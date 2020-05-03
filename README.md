# Gradient Descent Procedural Generation

### Background
A while ago I was involved in some discussions surrounding procedural generation. Specifically, how to make procedurally generated features "make sense". Generating terrain or features that conform to user expectations is important for user experience. For example, if a player was in a procedurally generated spaceship and told "There's a fire in the engine bay. Find it and put it out", the common player assumption would likely be that the engines are at the back of the ship. If the procedural generation algorithm has put the engines somewhere else on the ship, this may lead to player confusion and a worse gameplay experience.

### The Idea
At some point I had an idea: _What if we expressed the relationships of the various objects or components involved in the procedural generation in an objective function, and tried to minimize it?_ In theory, we would end up with a result that more-or-less conforms to the given constraints. As long as there is sufficient complexity and interdependencies in the constraints, and random initialization, we should settle in a different (hopefully valid) local minima each time we generate something, providing the diversity people typically look to procedural generation for.

This repo is my experiments with this idea.

## Blog
My thoughts and experiences throughout this adventure

#### May 1, 2020
Got something super basic working today. I'm optimizing the position and size of a lake, and the positions of 3 cities and my working example for now. The biggest issue I ran into for a while was that the optimization did not seem to be working. Sometimes it didn't seem to move anything in the right directions, or even move them at all. Turns out I was relying way too much on sigmoid functions, and was getting effectively 0 gradients when initializing random values because they would be so far out of range. Lesson learned, don't blindly use the loss functions used in Neural Nets just because they exist. This is a much different application. To prove to myself the optimization did work (at least somewhat) I used the negative absolute value to really force the variables to a specific value.

#### May 2, 2020
Today I planned on experimenting mainly with different cost/loss functions, but got sidetracked first enabling animations of the optimization progress #priorities. After being frustrated with `matplotlib` for a while, I found the [celluloid](https://github.com/jwkvam/celluloid) module which made things much easier. Now I can watch everything move around during the optimization and be confused (yet entertained) when it doesn't work. I should probably start writing unit tests...

I did still get the chance to mess around with some new cost functions. Overall I do really like the sigmoids and double sigmoids, because I do want certain ranges of values to "plateau" the cost. I just can't have plateaus outside the expected/desired ranges of these values. After reading some [clickbaity articles on loss functions](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0), I started using the log-cosh loss  to smoothly increase the cost outside the valid ranges of parameters. I also created a function that "interpolates" between points using sigmoid functions, so I can easily provide a list of where I want the peaks, valleys, and plateaus in the function to be, and get a more-or-less corresponding function (with log-cosh loss applied everywhere outside the range I provide).

While this new magic function does seem to be more-or-less working, cities now seem to end up inside the lake in quite a few cases, and other times everything just flies around quite aggresively without settling somewhere I'd expect. I wonder if it's settling because my loss functions are so stupid, the backtracking line-search ends up with such a small step value that we meet the relative or absolute score different termination criteria.

It also looks like when the lake and a city get close, they suddenly speed towards one another and overshoot such that the city ends up inside the lake or even on the other side. I think this might be because the gradients get quite large when they are close, and because this is "pure" gradient descent (not stochastic), both the city and lake move towards each other which results in a very large relative change. I'd like to try see if Stocastic Gradient Descent would help fix this by only moving one dimension at a time. Or simply putting a maximum on the step size for each dimension.
