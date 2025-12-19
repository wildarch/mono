/*
 * A reference implementation of the arithmetic expression evaluator.
 */
package main

import (
	"bufio"
	"fmt"
	"go/token"
	"go/types"
	"os"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	buf := make([]byte, 0, 10*1024*1024)
	scanner.Buffer(buf, 10*1024*1024)
	for scanner.Scan() {
		fs := token.NewFileSet()
		tv, err := types.Eval(fs, nil, token.NoPos, scanner.Text())
		if err != nil {
			panic(err)
		}
		fmt.Println(tv.Value.String())
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}
}
