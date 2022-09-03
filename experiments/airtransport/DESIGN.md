# Design ideas for an automated mass transport system in the air
Disadvantages of automated transport on the road:
- Deploying new infrastructure is incredibly capital intensive, and not possible in densely populated areas (cities).
- Using existing road infrastructure is being attempted with self-driving cars, but having to co-exist with human drivers makes things complicated. There are many difficult situations that autonomous systems struggle to deal with.
More and more cities are banning personal vehicles entirely from their centers.

Advantages:
- Air travel is easier to automate due to reduced obstacles
- Gliders do not need to store energy, which saves on weight. 
**Need to check if this actually is efficient**.

Goals:
- Be widely deployable, from rural areas to major cities.
- Competitive in terms of time needed to get from A to B. 
In particular, want to address the last mile problem that makes traditional public transport slow.
- Compact ground stations.
- Energy efficient.

## Launch
The idea is to launch single-occupant glider planes at high velocity, allowing them to gain enough altitude to make it to a nearby larger booster station.
One obvious problem here is the drag force on the glider, which increases with the square of the velocity.
To combat this the glider should have low cross sectional area and/or drag coefficient.

Seems similar to a winch launch.
A [paper simulating winch launches](http://www.pas.rochester.edu/~cline/FLSC/21179423-Numeric-Simulation-of-a-Glider-Winch-Launch)
provides some useful figures.

The glider considered in their paper is a [Schleicher ASK 21](https://en.wikipedia.org/wiki/Schleicher_ASK_21).
- Seats: 2.
- Weight: 360Kg.
- Wingspan: 17m

### Launch velocity
Our glider will be accelerating as quickly as possible, so we need to know what acceleration speed is tolerable.
Based on [Wikipedia 'g-force'](https://en.wikipedia.org/wiki/G-force#Horizontal), we will assume 20g is okay for a few seconds. This is a bit generous because it requires our passenger is sitting upright (lying down would be more aerodynamic).
We assume a 10m runway. This is a bit arbitrary. It seems small enough to fit in many locations (especially if placed vertically), but is still large enough to allow for some time to build up speed.

With 10m runway and 20g of jerk the launch sequence takes just 0.6s, but that is still enough to reach a velocity of 36m/s, or 130km/h.

We can also look at [rollercoasters with a fast launch](https://rollercoaster.fandom.com/wiki/Fastest_Launch_Acceleration) for an estimate. The record is held by [Do-Dodonpa](https://en.wikipedia.org/wiki/Do-Dodonpa).
Based on these numbers we could reach 172km/h in 1.6s, requiring 20.5m of runway.
If we go with a 10m runway instead, we will instead reach 18m/s or 65km/h.

The figures for both calculations are in the same ballpark, which is encouraging.
My purely theoretical numbers are probably a bit too optimistic, but equally it seems unlikely that a rollercoaster would have optimal performance, so let us pick some numbers that are towards the middle:
- Runway: 10m
- Exit velocity: 30m/s (108km/h).
- Launch time: 1s

## Flight distance
We will start with just launching the projectile at a 45 degree angle and ignoring drag.
That gives us an unimpressive 90m traveled.