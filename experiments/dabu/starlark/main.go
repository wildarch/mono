package main

import (
	"fmt"
	"log"
	"os"

	"go.starlark.net/starlark"
	"go.starlark.net/syntax"
)

func fibonacci() {
	// Execute Starlark program in a file.
	thread := &starlark.Thread{Name: "my thread"}
	globals, err := starlark.ExecFile(thread, "experiments/dabu/starlark/fibonacci.star", nil, nil)
	if err != nil {
		log.Fatalf("failed to execute file: %s", err.Error())
	}

	// Retrieve a module global.
	fibonacci := globals["fibonacci"]

	// Call Starlark function from Go.
	v, err := starlark.Call(thread, fibonacci, starlark.Tuple{starlark.MakeInt(10)}, nil)
	if err != nil {
		log.Fatalf("failed to call function: %s", err.Error())
	}

	fmt.Printf("fibonacci(10) = %v\n", v) // fibonacci(10) = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
}

func cmakeNinja(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	log.Printf("args: %s", args)
	log.Printf("kwargs: %s", kwargs[0])

	// TODO: Register target
	// - command to run
	// - dynamic/implicit dependencies?
	// - description
	// - inputs
	// - outputs
	return nil, fmt.Errorf("%s not implemented", b.Name())
}

func main() {
	thread := &starlark.Thread{Name: "my thread"}
	opts := &syntax.FileOptions{}
	_, err := starlark.ExecFileOptions(opts, thread, "experiments/dabu/starlark/build.star", nil, starlark.StringDict{
		"cmake_ninja": starlark.NewBuiltin("cmake_ninja", cmakeNinja),
	})
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
}
