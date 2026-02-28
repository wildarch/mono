# Build Systems
Build systems, especially cloud build systems (such as Bazel, Buck) are really interesting and useful software.
I want to investigate the following things here in more detail:
1. Are there opportunities for research? For this I will consult relevant (academic) literature
2. What does the design space for a cloud build system look like? What design choices are available when building a system from scratch?
3. If I were to implement a cloud build system, what should the design look like?

## Literature
The paper everyone points to [Build Systems à la Carte: Theory and Practice](Build Systems à la Carte: Theory and Practice).
I explored some of the works that cite this paper, but found little that was directly related to the design of cloud build systems.
The papers I found are in Zotero.

## Design Space
I base this primarily on the *Build Systems à la Carte* paper.

### Terms
- The purpose of a build system is to maintain the *store*: a mapping from keys to values.
  Typically, the keys are filenames, and the values are the contents of those files.
  In this case, the store can be implemented by a (directory within a) file system.
- A build system is *minimal* if it (1) only executes tasks that depend (transitively) on inputs that have changed since the previous build, and (2) it only executes each task once.
- *Dynamic dependency*: A dependency of a task that only becomes apparent once we have already started to execute the task.
- *early cutoff* describes an optimization.
  If a task `T` is executed because one of the inputs has been modified, but we find that the output of `T` has not changed, then we can avoid executing the tasks that depend on `T` (unless they have another input that has been modified).
- *shallow build* is a feature of cloud build systems where only the final build products are stored locally, and intermediate results are computed and stored exclusively in the cloud.

### Schedulers
There are three strategies for scheduling tasks, which differ in how they handle tasks whose dependencies are not ready yet:
1. Topological: Execute tasks in an order that guarantees the issue never occurs.
  This is only possible if are dependencies are statically known at the start of the build (or, after a separate analysis phase)
2. Restarting: Abort execution of the current task, and execute one of the dependencies instead.
3. Suspending: Pause execution of the current task, and resume it once the missing dependencies are available.

A suspending scheduler is theoretically optimal, but in practice it is only better than a restarting scheduler if the cost of storing suspended tasks is preferable over wasted computation time due to restarts.

### Rebuilders
A rebuilder decides which tasks need to be recomputed.
Four options for its implementation are:
1. Dirty bit: When an input is modified, it is marked as *dirty*. When a new build it started, all tasks that depend on *dirty* inputs are themselves marked *dirty*.
2. Verifying traces: When running an initial build, record the hashes of all values (initial inputs and task outputs).
   For a subsequent build, skip executing a task if the hashes of all inputs are the same (and reuse the existing task output).
3. Constructive traces: Like verifying traces, but also stores the outputs of each task.
   In a cloud build scenario with static dependencies, the central cache can index directly from the key and the hashes of its dependencies to the resulting value.
4. Deep constructive traces: Instead of looking at direct dependencies, look at the *terminal input keys* (terminal meaning the ground truth source files, which are the leaf nodes of the dependency graph). A clear advantage is that we only need to look at the terminal input keys, rather than exploring the dependencies recursively.

Two primary disadvantages of deep constructive traces:
1. Tasks must be deterministic.
   If not, if the results of an intermediate task are different from a previous execution due to non-determinism, this could be ignored because the terminal input keys have not changed.
   *NOTE: I don't think this is a real issue in my case, because we strongly want to avoid non-determinism anyway*
2. No early cutoff: we do not look at the intermediate results, so we must assume that all intermediate results could have changed.


Not up to debate for a cloud build system:
- Constructive traces: Early cutoff is important: Without it, changing a comment in one file can lead to a lot of unnecessary work.
  Buck has also switched to constructive rather than deep constructive traces, I assume for this reason.
  Dirty bit is not suitable for a system where multiple builds can be happening in parallel
  Verifying traces are effectively constructive traces without storing values, but we need those values for the cache to work.

Up to debate:
- Suspending vs. Restarting scheduler.
  Both are possible and there is precedent for both: Buck is suspending, Bazel is restarting.
