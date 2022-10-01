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

## Notes
Sqlite has various [security settings](https://www.sqlite.org/security.html) to prevent sandbox escapes through SQL queries.