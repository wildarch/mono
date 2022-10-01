# Functional Functions as a Service
I think a simple functional language would be perfect for implementing a Functions as a Service application.

Here's what I want from a FAAS for my personal usecases:
- Scripts receive an HTTP request, send back a reply 
- Scripts can send HTTP requests to other services
- Statically typed scripting language
- No build step for scripts (especially not full docker containers)
- Persistence using sqlite databases.
- Scripts must be able to run sqlite queries (and issue updates)
  By default every function gets their own, but we should also allow sharing

What we need to realize this:
- Frontend and type checker for our functional language
- Compiler and runtime for our functional language
- Standard library
- Backend server to route requests to the functions and create databases
- Optional: syntax highlighting and autocomplete in VS Code would be very beneficial 

Standard library requirements:
- JSON decode and encode
- HTTP request/reponse decode and encode
- HTTP client
- SQLite query and execute

## Language implementation
It probably makes sense to use ANTLR4 for parsing the language. 
I've used it for the 'Core' compiler and it worked very welll there.

The scariest bit is likely to be the type checker.
I think for the most part a classic a Hindley-Milner type system should do, 
but it seems like [extensions are needed](https://www.lesswrong.com/posts/vTS8K4NBSi9iyCrPo/a-reckless-introduction-to-hindley-milner-type-inference) to add support for Haskell-style type classes. 
This would allow you to write code that looks like:

```haskell
[0, 1, 2]
|> map (+ 1)
```

Instead of having to write the elm-style:

```elm
[0, 1, 2]
|> List.map (+ 1)
```

For the initial version though, elm-style is good enough.

For the backend we can start with a simple G-machine.
The most interesting thing there will be the implementation of an IO Monad (or similar).

## Notes
Sqlite has various [security settings](https://www.sqlite.org/security.html) to prevent sandbox escapes through SQL queries.