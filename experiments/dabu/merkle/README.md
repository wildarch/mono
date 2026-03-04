# Merkle Tree Builder
Creates a merkel tree of a directory.

Next steps:
1. Upload files and directories to a storage server
2. Use git to efficiently diff workspace

## Using git
Git can quite efficiently and very reliably determine which files in a repository have been modified.
We can use this to sync with the storage server.
Before uploading a full tree, we first check if the server stores a tree for the latest commit hash.
If yes, then we only need to upload the modified files.
The server computes the updated tree and returns the new hash.
