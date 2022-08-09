# Logistics algorithm
This is a quick analysis of algorithms for common logistical problems.

For starters, we will assume the role of a taxi company. We have at our disposal a set of taxis at specific locations, and wish to find the best way to service our customers by minimizing the time each of them needs to wait until a taxi can get to them.
To simplify a few things, we will assume that taxis know the exact time they are away from each customer, and that we have enough free taxis to service all customers.
For now we will also run the algorithm once for a static problem (i.e. nobody moves while we are computing, and no new customers arrive).

Given all of our assumptions, what we need is an assignment of taxis to customers that minimizes the total waiting time (or should we minize longest waiting time instead, which is more fair?). 

This problem is a classic, and called an unbalanced assignment problem. 

From reading the [wikipedia article](https://en.wikipedia.org/wiki/Assignment_problem), it seems the 'Hungarian Algorithm' is what we need for this.
An intuitive explanation is available on [Brilliant](https://brilliant.org/wiki/hungarian-matching/), but it glosses over a lot of the implementation details, like how to draw lines through rows and columns.
A more thorough tutorial with example code is available from [BRC](https://brc2.com/the-algorithm-workshop/)