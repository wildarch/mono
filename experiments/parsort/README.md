# Parallel External Merge Sort
We implement a tool that does the following:
1. Parse graph edges from a CSV file with lines formatted as `<row idx> <col idx> <float value>
2. Sort the edges by row and then column index.
3. (Optional:) Build a CSR out of them.

The parsing and sorting should be done efficiently, using multiple threads.
We assume that the system has `P` processors that can execute in parallel.
The strategy for parsing is:
1. Split the file into `P` blocks with an equal *number of bytes*.
   This very cheap since we can simply take the file size and divide by `P`.
2. Each processor parses one of the input blocks.
   We start parsing after the first newline delimiter to make sure we do not start parsing in the middle of a line.
   To compensate, we also parse until the first newline of the next block (for the last block this loops back to the first block).
   If we do not see a newline at the end of a block, we just continue in the next block.
   If the very last character in a block is a newline, then we start parsing a fresh line in the next block.
3. Parsed output is written to a per-thread temporary file.

For sorting, we do the following:
1. Sort the individual blocks, one block per thread.
2. Make pairs (A, B) of blocks
3. Get mid-point of A `mid = A[A.size / 2]`
   Find the corresponding split `(lo, hi)` in B such that all `lo[i] <= mid` and `hi[i] > mid`.
   Continue splitting until we have `P` pairs.
4. Merge the pairs.
5. Repeat from 2 until everything is sorted


