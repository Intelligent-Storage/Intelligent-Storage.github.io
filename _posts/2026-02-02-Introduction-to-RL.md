---
layout: post
title: "1 - Introduction to Reinforcement Learning and Human Behavior"
date: 2024-02-22
tags: [rl-modeling]
---

This writing experiment had first been intended to exemplify a rigorous approach towards dissecting current issues, exploring scientific mysteries, and thought testing ideas to a level that only written work can, for the ultimate purpose of improvement. I recently had the privilege of attending a conference which was both a change of routines, and an awareness refresher for my own circular behaviors. And that is this article's theme: understanding self-behavior.

It is not uncommon for self-help or self-improvement guides to make the connection between our behaviors and those that might be expected from a rat in various experimental paradigms. If we are not careful with how we conduct ourselves, these guides warn, we will be little different from our rodent friends who seek pleasurable rewards at every turn. The danger to be controlled by very basic stimuli is real. It is difficult to not have a relatable experience with this. And while realizing this connection for the first or second time can bring some kind of enlightenment, for me it never helped with understanding the larger behaviors at play that may not be preferable. Deeper research into literature for both *classical* and *operant conditioning* failed to illuminate much more.

But something clicked for me when I first encountered *Q-Learning,* a form of *Reinforcement Learning* (which is a form of machine learning). Ironic or coincidental, there is something to be said in the construction of machines and artificial intelligence to help us uncover the truths of ourselves.

> *"It is like mathematical equations. And each solved equation brings you closer to God. The act of creation is the intended use."*
>
> Anya Oliwa, Wolfenstein: New Order

I won't be taking any religious stances here, but if we consider 'God' to be the Universe itself, what it *really is*, then it could be said that engineering, mathematics, and research are the strongest forms of prayer.

Moving on. In this article, it is the aim to explain a basic model of *Q-Learning* so that later connections can be made to larger behaviors, although these should become self-evident. Although *Q-Learning* is generally explained mathematically, it is easiest to do conceptually, and would otherwise be inappropriate for this piece.

## Part 1: Choose Any Tile

Therefore I will introduce *Q-Learning* visually and with some basic definitions.

First we have the *universe:*

![The Universe](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.40.40-pm.png)

The *universe* encompasses everything in the visual above (and each of the visuals below). It is the entire world of our model. It contains all of the tiles, which are distinct locations. It also contains the *agent,* here the shown as the person symbol. The *agent* is an intelligence which is able to move from tile to tile, and in this case, represents ourself. The goal of the agent is to reach the *reward*, shown as the candy icon. The *agent* will proceed through the universe, to get to the *reward*. This is known as a trial, in which each trial begins with the agent at the starting point, moving from tile to tile, and ending at the *reward*. The *agent* will complete many trials, with the result of getting to the *reward* as efficiently as possible. Its experience in each trial run *reinforces its learning* so that it becomes better at locating the *reward*. The *reward* is just that, a location or item in the *universe* on a set tile which provides feedback to the agent.

Obviously in the model many arrangements and rules can be preset, but for simplicity our agent will always start on the *starting tile*, noted by the play symbol, and the *agent* can move to any adjacent tile. The *reward* will always be on the same *reward tile.* The *agent* can't see what is in any of the tiles until it has entered that tile, so it can't see where the *reward* is, even if it is beside it. If it helps conceptually, imagine each tile is an empty, closed 4-wall room, and if there is an adjacent tile there will be a closed door to open on the corresponding wall. When the *agent* moves to an adjacent tile, it passes through one of the doors. The only exception is the reward tile which would be a room having the reward.

The last definition is for *Q-values*. These are values our *agent* attributes to tiles (locations) at the end of each trial if it has crossed that tile in the past. These values represent a sort of connection to the reward which will become evident as the example progress. The *reward* tile always has a value of 1.0, and this is the highest value that no other tile can ever receive.

As previously said, there is a looping of trials. At the start of each trial the *agent* will begin on the *starting tile* and move randomly from tile to tile until it finds the *reward*. Then understanding the path the agent took, it will update the *Q-Values* of those tiles. A new trial will begin. Eventually, given its understanding of the board, the agent will find the shortest path (or one of the shortest paths) to the reward.

Here is Trial 1:

To start, the *agent* will randomly pick a tile. Here it moves 1 tile right. The blue shows the highlighted path:

![Trial 1 - Step 1](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.51.05-pm.png)

Again, it will continue to choose random tiles as it moves along, until it finally reaches the *reward.* By this point the agent has moves 2 tiles up.

![Trial 1 - Step 2](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.51.09-pm.png)

Now it has moved 3 tiles to the left, then 1 tile up

![Trial 1 - Step 3](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.51.13-pm.png)

Now 1 tile left, and 2 tiles up

![Trial 1 - Step 4](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.51.16-pm.png)

And finally 4 tiles right, reaching the reward

![Trial 1 - Step 5](/assets/images/rl-behavior/screen-shot-2024-01-20-at-6.51.19-pm.png)

The process will start anew this time except the *Q-Values* will be updated:

![Trial 1 - Q-Values Updated](/assets/images/rl-behavior/screen-shot-2024-01-20-at-7.12.52-pm.png)

Now the *agent* is faced with a dilemma. Does it follow the path it now knows leads to the reward? Or does it choose to explore new paths?

The *agent* does not know the shortest path to the *reward* is only 7 steps. It does know its first path is 13 steps. The only way to gauge this path is by another trial, but this risks taking an even longer route at the risk of wasting time to get a reward. Imagine for instance that our *agent* only gets 5 minutes to gather as many candies as it can, and moving between tiles requires 10 seconds. Exploring could cost it time which could otherwise be spent gathering rewards. But exploiting the only known path may be inefficient and a shorter path could be found allowing the *agent* to then gather more rewards in a shorter period (which we know as outside observers to be true in this case).

We will explore this question, classically known as "Exploration vs Exploitation" and its connection to human behavior in the next section.

## Part 2: Exploit or Explore

This section explores how it is possible to understand complex human behaviors with simple Reinforcement Learning, a type of Machine Learning. Specifically Q-Learning. The previous section covered concepts in a basic Q-Learning model: *the universe*, *the agent*, *the reward*, and *tiles*. This section will cover the iterative *reinforcement* process, what *Q-Values* are, and the question of *Exploitation vs Exploration*. Ultimately, this is the final set-up for a discussion on modeling human behavior.

But briefly, I will digress to explain a term that has wrongly been assumed to be understood: *model*. What we really are talking about here is modeling human behavior. What is a model then? In science, the term *model* is used to denote a representation for a system (ie a grouping of connected factors). Just like toy models, such as dolls or train sets, scientific models are simplified representations of the real thing which enable us to better understand the phenomenon and allow for accurate, but *not perfect* predictions. Something I struggled with for a long time was understanding why models are *not meant to predict with 100% accuracy* (excepting the simplest artificial systems). In fact, if a model predicts perfectly this is *bad*. A bit counter-intuitive. Isn't what we want is to make perfect predictions? To know exactly how much snow is going to arrive, what the future cost of a house or profit on an investment will be, what decision someone will make? While that is the *ideal,* any model that does this fails on at least one of two conditions.

In order to achieve perfect or near-perfect accuracy, either a model has incorporated too many variables, becoming too complex and unwieldy for intuition (the express purpose), or it is so perfectly fitted to a given group of observations (ie a dataset) that it no longer generalizes to other observations of the same phenomenon outside the originals. In short, a good model needs to be relatively accurate, simple enough to understand, and generalizable to future observations (and historic ones as well).

In the last section, we paused our Q-Learning model after the first iteration, in which our agent took a random walk through tiles and eventually found the reward after 13 steps. After this journey, the agent updated the *explored* tiles with *Q-Values*.

![Q-Values after first iteration](/assets/images/rl-behavior/screen-shot-2024-01-20-at-7.12.52-pm.png)

There are two questions which presented themselves:

1. What are *Q-Values*?
2. In the second iteration, what *should* the agent do?

### What are Q-Values?

In our context, *Q-Values* are the *expected reward* an agent has for a given tile. At the beginning, the agent had no preconceptions of the value of any tile. But now after an initial exploration and discovery of the reward location, the agent has realized a connection between explored tiles to the reward. There is no *intrinsic values* for the tiles themselves (beside the reward tiles), so *Q-Values* can be thought of as a sort of set of anticipated internal rewards.

The reader will note that the *Q-Values* do not uniformly increased across the tiles. At the beginning, *Q-Values* only augment by 0.05 per tile, while the last tiles experience increases of 0.1 and 0.2. There are different ways the *Q-Values* can be updated after an iteration, and in this case a *discount factor* is applied. The "farther" a tile is from the reward, the greater the *Q-Value* will be discounted. In this case "farther" is not the true distance but the *perceived distance*. In short, tiles which are *believed* to be closest to the reward experience the least discount, we value them more, and those *believed* to be furthest receive the greatest discount. Why is this the case?

There are several ways to conceptualize this, but here is a simple one: the agent is surest about the tiles it took closest to the reward, while those taken further away it is less certain about. Tying this momentarily to human behavior and reward, we are far more certain that the *action* of opening a bag of cookies will lead to getting cookies, the reward, than we are to the *action* of entering the the kitchen. After all, maybe the cookies are hidden or we forgot them in the car and they're not in the kitchen at all.

This hints at something I will come back to later. But here, *tiles* are shown to be *locations*. Really, a *tile* is a representation for an *action* in a given state.

In summary, *Q-Values* are non-real, expected rewards for a given action (i.e. movement to a given tile). They are based on *experience*. *Q-values* can be assigned in various ways. One common method is using a discount factor, depreciating expectations relative to their perceived distance from the reward.

### Explore or Exploit?

Now we tackle the question of iterations: What should the agent do given that it knows a path to the reward? Out of context, this question doesn't have a meaning. Without external pressures that affect the agent there is no reason to (or not to) repeat its initial behavior.

But the real world doesn't host ideal environments. There are pressures in every way. Among the infinite other pressures, it is uncertain if the reward will persist indefinitely or if it degrades with time, it is uncertain if the agent is competing against other agents, and it is uncertain how often the agent must find the reward to stay alive.

> "…but, in this world, nothing is certain except death and taxes."
>
> Benjamin Franklin, Letter to Jean-Baptiste Le Roy, 1789

Given a pressure, for example, that the reward will only last 60 seconds, how should the agent act to *maximize* rewards overall? First, it might be helpful to grasp what the limits are in this case: For a given 60 second test, what is the worst, the best, and the current time it takes to find the reward?

Let us consider that each movement between a tile requires 2 seconds, and each iteration reset requires 1 second. Let us also assume in the calculation below the agent is allowed the first explorative iteration of 13 steps before the 60 second trial begins.

*One* of the best case scenarios is shown below. There are multiple best case, each requiring no more than 7 steps.

![Best case scenario](/assets/images/rl-behavior/screen-shot-2024-01-26-at-3.30.21-pm.png)

In this case, the total time required for each iteration is:

7 steps × 2 sec + 1 sec = 15 seconds

In 60 seconds, this means the agent can find 4 rewards, at most.

One of the worst scenarios is below. Admittedly, I didn't verify that this is the worst possible path, however, it requires 33 tile movements which means that within the 60 second period, the agent will never reach the reward:

![Worst case scenario](/assets/images/rl-behavior/screen-shot-2024-01-26-at-3.42.39-pm.png)

In this case, the total time required for each iteration is:

33 steps × 2 sec + 1 sec = 67 seconds

As already mentioned, the known path requires 13 steps:

![Known path](/assets/images/rl-behavior/screen-shot-2024-01-26-at-3.44.59-pm.png)

In this case, the total time required for each iteration is:

13 steps × 2 sec + 1 sec = 27 seconds

For 60 seconds, this means the agent can find the reward 2 times.

In summary, with its current behavior, our agent will receive the reward twice in the given time period. At best the agent could improve this to 4 rewards, but at worst it will receive no rewards. And this is the heart of the problem: Should the agent decide to *explore* instead of *exploit* its current knowledge, it is unknown whether the new knowledge will benefit the agent or cost it time. Given the probabilistic choices, there is always a chance for either outcome.

An astute reader might consider it is possible to calculate the likelihoods of choosing by chance a 7, 8, …, and X-step pathway to the reward given a random walk. Given the close proximity of the starting tile to the reward tile, it even appears without such an analysis it is far more likely to choose a shorter walk than a longer one. But such analyses miss the point that to the agent this is *external information,* unavailable to it. We could obviously advise the agent given our view point, but then why not push this limit of advice towards directing it to the reward?

In this model, just as in life, it is difficult to determine the limits of the action space available to us, and just as the agent is blind to the location of the reward, we too are blind to the variety of rewards that may exist.

The general solution to the exploration vs exploitation question is a tradeoff balance. Ultimately, the probabilities, universe characteristics, and initial experiences will always play a significant role in the success of the strategy employed. On one hand, the agent can be *greedy* which means that given a known solution it will tend to stick to that solution. On the other hand, it can be explorative and explore more than it exploits. The propensity to exploit or explore can be determined by a variable, which we may call *greediness*. We can assign it a probability such that at 0.50, a given iteration has equal chances of either repeating the best known iteration or choosing to explore. At 1.0 *greediness,* that is *maximum greediness*, the agent will always exploit the best known pathway.

Our timed example has very stringent requirements. Even with the best known pathway, the agent only gets to iterate over the given universe 4 times. At the worst pathway, only once (and never actually finish). In typical reinforcement learning, hundreds to many thousands of iterations take place, and in comparison with our own lives, we repeat our behaviors just as many times. Consider, just how many times have you practiced saying the word "the"?

Given the agent's initial experience in our example, it might be considered acceptable to have an extremely greedy policy given a working solution. Why risk exploring and getting no reward when 2 rewards are guaranteed? But what if our time limit extended to 60 minutes, not 60 seconds? The best path would result in 240 rewards, and our current path only 133. In this case the pressures appear lenient enough on the agent that a more explorative behavior will have a veritable opportunity to provide higher returns in the long run.

It must be noted that our example is very, very simple. In realistic reinforcement learning models, it must be considered that explorations need not be totally random, and there exist smart algorithms to help the agent navigate to the reward faster. In one algorithm for instance, the lower the Q value of a given tile, the greater the agent might be encouraged to explore. Whereas, when the agent is on a tile near to the reward, which due to the lower discount factor will have a higher Q value, the agent is less likely to explore because the certainty of obtaining the rewarding already quite high.

### The Iterative Process and Updating Q-Values

The best way to further explore reinforcement learning and how it can be used to model human behavior is through iterating over the trials so see how this plays out in the long run. Let us assume on the next trial the agent decides to exploit, and takes the same pathway as before:

![Exploiting the known path](/assets/images/rl-behavior/screen-shot-2024-01-26-at-4.49.37-pm.png)

How does the agent update the Q-Values? Do they even change since we already were aware of this path? Yes, they do.

Now that the same pathway has been taken with success again, the agent has a higher certainty this is a path which leads reliably to the reward. Remember, our example is very simple, but even small changes can cause major effects. For instance, imagine if on every third trial, the reward changes position. Then the expected path way may no longer be as efficient, and possibly far from it! But we will stick with a simple, static reward. Because there were already previous non-zero Q-Values, the values will change but not as strongly as if the initial estimate was zero. Remember, Q-Values are anticipated rewards. In the first iteration, the agent has no expectation the taken tiles would lead to a reward, wheraas now some expectation already existed:

![Updated Q-Values after exploitation](/assets/images/rl-behavior/screen-shot-2024-01-26-at-4.54.57-pm.png)

Now imagine on the next trial, the agent explores and takes the following path:

![Exploring a new path](/assets/images/rl-behavior/screen-shot-2024-01-26-at-4.56.41-pm.png)

How do we update the following Q-Values, and take account of the previous Q-Values on tiles not explored on this iteration? Again, there are many solutions to this but one way is to update the path taken as normal, and apply a *forgetting discount* to all of the tile *not taken* during this iteration. By applying a *forgetting discount* to untaken tiles, our agent takes into account the most recent experiences of the agent (remember the universe can change), but also makes better knowledge more likely to become pertinent in the face of an excessively greedy policy. In this case, an even 0.05 decrement of the Q-values is taken across all untaken tiles to "forget" them.

![Q-Values with forgetting discount](/assets/images/rl-behavior/screen-shot-2024-01-26-at-5.00.27-pm.png)

### What will happen in the long run?

Below is a hand-made (artificial for an artificial system!) plot of what the expected Q-Values might be after many iterations, with a somewhat ambivalent greedy policy (typically exploit, but some times explore). The settled path is marked in blue. We observe that at each possible step, the agent moves towards the subsequent tile with the highest Q-value.

The agent has identified one of the best routes, and it follows this route at least half the time. It also has high Q-scores along many of the other shortest paths. And the agent has also identified the tiles which are not very likely to get it to the reward in the most efficient manner.

(There is one mistake tile, can you spot it?)*

![Long-run Q-Values](/assets/images/rl-behavior/screen-shot-2024-01-27-at-6.45.41-pm.png)

*Mistake tile is the one to the left of the starting tile.

### So…What's the point?

There are two key take aways from this section:

1. Given an appropriate greediness policy, and enough time, an agent will converge on an optimal sequence of tiles (ie actions)
2. With enough exploration and trials, behaviors that work but are not optimal will be pruned (ie forgotten)

The point of this rather large build-up to introduce Reinforcement Learning (Q-Learning variant here) and its connection with human behavior will be explored in the following sections.

## Part 3: But What About the Real World?

*"Inside of me there are two dogs. One is mean and evil and the other is good and they fight each other all the time. When asked which one wins I answer, the one I feed the most"* - Attributed to Sitting Bull

In the last section on Reinforcement Learning and Human Behavior, I covered the question of "exploration versus exploitation," the problem of deciding to follow past successful behavior or to search the environment for other possible ways to succeed. This included understanding that this behavioral decision is context driven, and as a rule, there are no true solutions, only trade-offs. A balance between exploiting and exploring must be reached. With such a balance, given enough time (and lack of hard pressures like death) the agent will arrive at an optimal or near-optimal solution.

Something I forgot to discuss in detail in the previous section is that this decision to explore or exploit can also be time dependent. In the example from before, we considered a single variable *greediness* that was equivalent to a coin toss on each trial. The result of this coin toss dictated whether the agent exploited, went for the reward on its best known route, or explored, seeking the environment for a reward in a non-optimally believed route. Consider however that this *greediness* variable can itself vary with time. In an environment which does not change, or changes little over long periods, it is best to be most explorative in the beginning and then very greedy later after the environment and most of the paths (seem to) have been explored. Otherwise, high greediness from the start is less likely to find the optimal path in the long run because fewer avenues will be explored. To explain this more technically, consider this: the chances of selecting the optimal path is the number of optimal paths over all possible paths. Of course, even if the agent selects this optimal path on the very first run (a very low possibility), it has no knowledge as to if it is the optimal path. The chances of the agent finding the optimal path is therefore:

![Chance of finding optimal path equation](/assets/images/rl-behavior/screen-shot-2024-02-11-at-5.07.34-pm.png)

I only bring this very simple math equation here to highlight that the factor of 'Paths Explored' clearly impacts the chances of Finding the Optimal Path (albeit in a simplified example). But each time the agent takes a new path is like collecting a new lottery ticket number, and the more lottery tickets one has, the better the chance of winning.

Of course, not all environments are static or change slowly. Environments themselves can vary with time, sometimes changing greatly over the given period, or changing slowly most of the time, then a whole lot all at once, or sometimes the environment flip-flops between cyclic states (as described earlier where every 3rd trial the reward might change tiles). Therefore, the way *greediness* should vary depends on the how the environment changes. The above equation of the Chance of Finding the Optimal Path does not hold in a constantly changing environment. If the agent stopped exploring in a changing environment, it will soon find the optimal path is not so optimal.

With this overall general framework for Reinforcement Learning, it's time connect it with concrete examples in the real world. Perhaps the easiest introduction to this is by revealing the closest approximation for the earlier, simple Reinforcement Learning example. This example is akin to a laboratory mouse in a maze, looking for a piece of reward cheese. In fact, it's almost an idealized version of it.

Here's an example diagram of such a mouse maze overlayed on the example. Dark blue lines are walls.

![Mouse maze overlay](/assets/images/rl-behavior/image.png)

Just as in our example with the agent, the first time a mouse enters the trial maze it will have no previous knowledge on how to navigate the maze to find the reward. Of course, in the real world, depending on how the maze is designed and used and what the reward at the end may be (food, water, an exit, etc.), various factors might influence the initial behavior.

For instance if the reward is a smelly cheese, this might give some general direction to the mouse who has a keen nose. Although, it can easily be imagined that in certain maze designs, such as this one, the smell could draw the mouse physically closer to the reward, but further from the path that accesses that reward. In this case, the smell of cheese would attract the rat to two potential dead-ends which would never let it get the cheese, while the path which goes furthest from the smell is that which leads to the reward. Smelly cheese is not the only possible factor: lighting, smell of mice that may have previously passed through the maze, and so on. But generally speaking, on the first run, the mouse is a *tabula rasa*. And only after a series of repeated runs through the maze will the mouse become accustomed to the best path, that which allows it to access the reward in the fastest time and with the least amount of movement (i.e. avoiding false corridors or passages).

It can be said that this is not a true real world example, for after all, the whole maze is an idealized real world situation. Where in the world are there straight corridors, perfect corners, square dead ends, and a little piece of cheese at the end, all in mouse size dimensions? Well, no where. But we have gone from a completely theoretical model, to a real-world model, and now we can consider that this also extends to the real world.

The framework for Q learning which has been presented can be extrapolated to many other problems, big and small. Instead of tile placements representing spaces, again consider each tile as an action. One of the earlier examples was to remind yourself how many times in your life you've said the word "the". Each time you've said this word, you are in a sense running through the maze of matching the correct motor control of your mouth and tongue in order to arrive at the "best" pronunciation of "the."

Anyone who has had small children of their own or been around them while they are in the cusp of speaking their first words will realize (if they did not already) that baby babbling isn't nonsense. It is the equivalent of the mouse moving through the maze. The child babbling is them slowly understanding the associations between different tongue positions, their timing, and breath controls to various sounds, and slowly matching these sounds to those in their maternal language.

Try and say the word "mom" and count the number of identifiable steps going on in your mouth and breath in order to pronounce this one word. Then, even for the first couple steps you identify, consider all of the other possible actions that you could have done instead. These are all the other tiles (actions) that could have been taken. Consider that this one simple word which we often look for as one of the first words out of a child mouth, probably took your very self over a year to learn to pronounce. And you almost certainly didn't do it very well at first either. This example primarily demonstrates a motor problem (i.e. how to move parts of the body), but we can extend this to larger human behaviors.

Consider for instance how you first learned to do a handshake. Or ride a bike. How to tie your shoe. All of these require a series of progressive steps in order to achieve success. For each of these examples, how many tries did you require before being succesful? Considering handshakes, how long before you could do a non-awkward one (if you've managed), and consider this, how many awkward handshakes do you still experience at this point in your life?

"Everything is hard before it is easy" – Attributed to Johann Wolfgang von Goethe

Something to remember is that for each of these tasks, the learning process never ends. Each time you are exposed to these problems, and others like them, the world and yourself have always changed a little, and we are always striving for optimality.

While we have only considered rewards so far in our Reinforcement Learning example, it is now time to introduce the anti-reward: punishments. These are objects, situations, etc. in short stimuli that we (and agents) want to avoid. In the below example, it is the smiley face.

![Punishment introduced](/assets/images/rl-behavior/image-1.png)

This is pretty straightforward concept (and should've been introduced earlier).

## Part 4: Is This Really Worth Learning About?

After reading all of this, one might think: that was a lengthy explanation for what could have been said in a few sentences: "There are things that people like and dislike. When people find a way to get the things they like, they will repeat that behavior, and the opposite for things they find displeasing." One might even go so far as to include, "And it's important to sometimes explore sometimes, because that's how people find new things they like or better ways to get the things they already like."

And all of this is true. But this summarization misses the purpose of a model because it lacks the information that makes a model useful. What are the two main purposes of a model? To predict and to allow for better understanding. The summarization might seem to do that, but it lacks the *how.* People might repeat behaviors that lead to a previously pleasing experience, but how does that happen? How does this summarization give us insight to the underlying process?

Exploring concrete examples which illustrate this *how* will make clear why it is worth the effort of understanding reinforcement learning. The first case we will consider are addictive or "unwanted" behaviors. It is easy to conceptualize how such behaviors arise, both by the summarization above, but also from our simple RL example. Imagine once again we return to our initial agent who already has found an optimal path, but sometimes explores:

![Agent with optimal path](/assets/images/rl-behavior/image-2.png)

Then, at some point, a new "reward" enters the environment:

![New reward appears](/assets/images/rl-behavior/image-3.png)

Recall, the agent is blind to changes but eventually given enough time, chance, and an explorative behavior it will find the new reward. In this case, the new "reward" is much closer and easier to get, requiring only 4 steps. Not only that, but maybe the value of this reward is even higher than the typical reward. The result is that the agent begins to take a shortcut (shown in red) and since the maximum value that can ever be a experienced is a value of 1, the original reward's value is normalized, meaning it becomes lower in this case, say 0.75. It no longer has the same 'oomph.'

![Agent taking shortcut to new reward](/assets/images/rl-behavior/image-4.png)

The summarization can explain this change in behavior, "When people find a way to get the things they like, they will repeat that behavior…," but the model clarifies several external observations, and makes predictions. First, we gain an understanding of how an extreme stimuli (i.e. heroin) can make the everyday pleasures muted. In this case, the original reward, say a piece of candy, lost 25% of its value. This is because the most reward the agent can ever experience is limited to a maximum of threshold. Should new rewards be introduced, all the values are normalized, meaning they are all scaled the the highest value is no greater than 1.0. Second, we see why the agent began to display addictive behavior. Compared with the original reward which required 7 moves, this new reward only requires 4, in addition to the newer reward having a greater relative value. If we were to run this through the 60 second trial, it could receive the new reward 6 times (2 times or 50% more than the previous optimal route). And finally, and the most importantly for this piece, is that the model helps us understand how to best break out of an addictive or unwanted behavior.

The worst thing an individual can do when they have recognized they are in the sequence of events which completes an addictive behavior is to complete it. It of course may come into the mind of any individual, "This will be the last time, then I'll stop," or "Just this last one." Most of us can recognize this, anything from hard addictions to drugs, to trying to improve our diets and cutting out junk food, to escaping the doom scroll. When an agent (or you) complete that "one last time," nothing is gained except *reinforcing* the same behavior. It only makes it more difficult to quit the next time because the memory of how to achieve this reward just got stronger.

The best method for escaping an addiction is to break out of the behavioral pattern once it has started. Breaking out of the path, like choosing a new tile and a new route, weakens that "addiction shortcut". And at the same time it strengthens a new one.

Besides addictions, our reinforcement learning model helps us to explain why people struggle to create new habits or try things in a new way, which the summarization does not. According to the Q-Model, there is an "inherit" reward in doing things the same way (i.e. selecting the same tiles). In our model, the Q-values, although not a real rewards, are the anticipated or expected reward of doing the action. Every time you begin a behavior you are accustomed to doing, there is a chain of anticipated rewards for each step. To stop, or to start practicing a new route is difficult because the anticipated reward suddenly drops. This may seem "woo" but research has indicated that among individual temporary given control of intracranial stimulation, they are most likely to active the region which does not in fact produce sensations of pleasure, but the sensation of anticipation [citation needed].

Models aren't perfect. Q-Learning is not even generally mentioned as a method for describing complex human behaviors.

### Things that Q-Learning and this model have not covered:

1. We have not included discussions on neuroeconomics, and what is the "cost benefit" analysis of any action.

   > "Economics is the study of the use of scarce resources which have alternative uses."
   >
   > Thomas Sowell, Basic Economics, 2000

   For instance, some of us at one time or another might have had the experience of putting something off, which once we finally begin, we may continue to do for a lengthy period of time beyond that which we initially agreed with our selves. This paradoxical change is not explained in our model, because we ourselves know we are bound to repeat this same behavior. Familiar examples might include practicing an instrument, cleaning, and yard work.

2. We have not explored what exactly constitutes a "reward" or a "punishment". How does the agent define these exactly? Why is it that in humans, the same stimuli might be a "reward" to one, but a "punishment" to another? Why are rewards normalized?

3. How is it that addiction relapses happen? This is perhaps a fault with such a simplified model, although I'm not personally aware of how this is implemented in more advanced ones. In our example, there is an equal forgetting over time, so eventually the addiction shortcut is forgotten. This is not normally the case for most addictions, for even years after avoiding the addictive reward, a single new consumption can instantly reignite that addictive behavioral pattern (though this is not always the case, and hard addictions are a complex topic in themselves).
