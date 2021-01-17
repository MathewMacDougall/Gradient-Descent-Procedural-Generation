# Gradient Descent Procedural Generation

### Background
A while ago I was involved in some discussions surrounding procedural generation. Specifically, how to make procedurally generated features "make sense". Generating terrain or features that conform to user expectations is important for user experience. For example, if a player was in a procedurally generated spaceship and told "There's a fire in the engine bay. Find it and put it out", the common player assumption would likely be that the engines are at the back of the ship. If the procedural generation algorithm has put the engines somewhere else on the ship, this may lead to player confusion and a worse gameplay experience.

### The Idea
At some point I had an idea: _What if we expressed the relationships of the various objects or components involved in the procedural generation in an objective function, and tried to minimize it?_ In theory, we would end up with a result that more-or-less conforms to the given constraints. As long as there is sufficient complexity and interdependencies in the constraints, and random initialization, we should settle in a different (hopefully valid) local minima each time we generate something, providing the diversity people typically look to procedural generation for.

This repo is my experiments with this idea.

#### Blog / My Progress
My very first few commits here were from tinkering at the end of my school semester when I had some time. It was quick and dirty, but did seem to show the idea worked. I got busy with other things immediately after and didn't get back to this for a while. Here I am 9 months later (older and wiser) ready to try this again for real.

I scrapped all my previous code because it had no real structure. I wrote my own simple versions of Gradient Descent and Stochastic Gradient Descent. I then made a simple class to describe each "optimization problem" for the procedural generation. Sprinkle some matplotlib in there for visualization, and it was good to go. With all that ready, I made a very simple example that optimized the position of a single point. This worked as expected, but isn't that interesting so I'm not going to show it here.

Now that I had the simplest possible proof-of-concept working, I moved on to try make a slightly more complicated example: Placing 3 cities around a single body of water (which I refer to as a lake).

The position of each city, the position of the lake, and the radius of the (perfectly circular) lake would be the optimization parameters. The cost function was still relatively simple:
* The lake radius was encouraged to be a specific radius
* Cities try be near the water
* Cities try be far from one another

Before we talk about the issues this exposed, lets appreciate this cool animation:

TODO: animation here

Now although this barely more complicated example worked, it did expose some drawbacks with the current setup:
1. Setting up the optimization vector and converting between the vector and more useable datatypes is very verbose and tedious. This makes it more difficult to initialize the vector, and be concise and expressive in the cost function. If it was annoying with only 9 values in the vector, it will only get exponentially worse for larger problems.
2. Writing cost functions is difficult and error-prone. My very first version of the cost function had the following line: `lake_cost = lake_radius^4`. This caused the optimization to silently fail and hang the program, because the gradient was so large that even at moderate values of `lake_radius`, the gradient descent algorithm was unstable and unable to converge. Changing this line to `fabs(lake_radius - 2)` solved this problem, but clearly indicated the need for a better way to write cost functions such that issues like this can be avoided.

