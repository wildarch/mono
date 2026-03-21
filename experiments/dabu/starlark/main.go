package main

import (
	"fmt"
	"log"

	"go.starlark.net/starlark"
)

func main() {
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
