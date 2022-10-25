# SKI Combinatory logic compiler and runtime for Core

This folder contains a compiler and corresponding to runtime for the Core language based on the S, K, and I combinators.
The implementation is based on the
book [The Implementation of Functional Programming Languages](https://www.microsoft.com/en-us/research/uploads/prod/1987/01/slpj-book-1987.pdf)
,
with aspirations to eventually incorporate the ideas
of [Lambda to SKI, Semantically](http://okmij.org/ftp/tagless-final/ski.pdf).

Some links that seem useful:

- https://github.com/Superstar64/sky
- https://crypto.stanford.edu/~blynn/lambda/sk.html

## A basic SKI Compiler

For now, we will not worry about performance or being feature-complete, a basic working SKI compiler is good enough.
We will be skipping operations on primitives (e.g. addition, multiplication or integers), case distinctions,
constructors and let bindings.
This lets us fully capture a function definition in the following data type:

```kotlin
sealed class Comb

data class CAp(val func: Comb, val arg: Comb) : Comb()
data class ConstInt(val n: Int) : Comb()
data class FuncRef(val name: String) : Comb()
object I : Comb()
object S : Comb()
object K : Comb()
```

The full program is a map of functions defined in terms of `Comb`:

```kotlin
typealias CombProgram = Map<String, Comb> 
```

If we were going for maximum efficiency, we could store the functions in a list instead and refer to them using indices,
but that is annoying to debug, so we will not bother.

Compilation follows algorithm 16.2 from 'The Implementation of Functional Programming Languages' (page 264, or 276 in
the PDF).

```kotlin
private fun compileC(expr: Expr): Comb {
  return when (expr) {
    is Ap -> CAp(compileC(expr.func), compileC(expr.arg))
    is Lam -> {
      var exprC = compileC(expr.body)
      for (param in expr.params.reversed()) {
        exprC = compileA(param, exprC)
      }
      exprC
    }
    is Num -> ConstInt(expr.value)
    is Var -> when (expr.name) {
      "I" -> I
      "K" -> K
      "S" -> S
      else -> FuncRef(expr.name)
    }

    is BinOp -> TODO()
    is Case -> TODO()
    is Constr -> TODO()
    is Let -> TODO()
  }
}

private fun compileA(param: String, expr: Comb): Comb {
  return when (expr) {
    is CAp -> CAp(CAp(S, compileA(param, expr.func)), compileA(param, expr.arg))
    is FuncRef ->
      if (expr.name == param) {
        I
      } else {
        CAp(K, expr)
      }

    else -> CAp(K, expr)
  }
}

fun compile(program: Program): CombProgram {
  return buildMap {
    for (def in program.defs) {
      put(def.name, compileC(Lam(def.params, def.body)))
    }
  }
}

```

**TODO**: Explain Lam abstraction of multiple parameters

## A basic virtual machine

We define an equally primitive virtual (some papers call this 'abstract') machine to execute our programs.
Let us start with the type to represent the machine state.
Besides storing the program, it also keeps track of the current combinator we are reducing, and we will add a stack to
park combinators to be reduced later (we will see where this is needed shortly).

```kotlin
data class SkState(val program: CombProgram, val comb: Comb, val stack: List<Comb>)
```

Evaluation proceeds by creating the initial state, and then repeatedly applying reduction steps, until we reach a final
state (where nothing can be reduced further).
Any small bug in our evaluator can make it loop forever, so let us add a maximum number of steps to ensure it always
terminates.

```kotlin
fun evaluate(program: CombProgram, maxSteps: Int = 10000): List<SkState> {
  val trace = mutableListOf(SkState(program, program["main"] ?: error("missing main"), emptyList()))
  var steps = 0
  while (!isFinalState(trace.last())) {
    if (steps > maxSteps) {
      error("Did not terminate after $maxSteps")
    }
    trace.add(step(trace.last()))
    steps++;
  }
  return trace
}
```

The `step` function applies a single reduction step to the current state and returns the resulting new state.
Here is what we will do for each combinator:

- `CAp`: While there are only three combinator that `arg` can be applied to, `func` does not need to be plain
  combinator (yet).
  It could be a very complex expression, all we know is that eventually that expression resolves to either S, K or I (
  technically S and K take more than one argument, so it might also resolve to something like `CAp(K, *))`).
  We will need to evaluate `func` and see what it reduces to before deciding what to do with `arg`, so we will `arg` on
  the stack and continue reducing `func`.
- `FuncRef`: We have the definition of each function stored in the state, so we will look up the combinator definition
  for the given function, and substitute it for the reference.
- `I`: This function returns its argument unchanged. The argument was previously pushed onto the stack by `CAp`, so we
  will pop it and set it as the current combinator.
- `K`: Takes two arguments and returns the first. Consequently, we will pop two arguments off the stack and set the
  first popped element as the current combinator.
- `S`: Applies an argument `x` to combinators `f` and `g`. We instantiate it by setting the current combinator
  to `CAp(CAp(f, x), CAp(g, x))`.
- `ConstInt`: We cannot reduce an integer any further, it marks a final state.

The `isFinalState` function is thus easily defined as:

```kotlin
private fun isFinalState(state: SkState) = state.comb is ConstInt
```

And finally, our `step` function:

```kotlin
private fun step(state: SkState): SkState {
  return when (val comb = state.comb) {
    is CAp -> state.copy(
      comb = comb.func,
      stack = state.stack + comb.arg
    )
    is FuncRef -> state.copy(
      comb = state.program[comb.name] ?: error("missing function ${comb.name}")
    )
    I -> state.copy(
      comb = state.stack.last(),
      stack = state.stack.dropLast(1)
    )
    K -> state.copy(
      comb = state.stack.last(),
      stack = state.stack.dropLast(2)
    )
    S -> {
      val f = state.stack[state.stack.size - 1]
      val g = state.stack[state.stack.size - 2]
      val x = state.stack[state.stack.size - 3]
      state.copy(
        comb = CAp(CAp(f, x), CAp(g, x)),
        stack = state.stack.dropLast(3)
      )
    }
    is ConstInt -> error("Cannot reduce int further")
  }
}
```

TODO: S-combinator duplicates arguments.