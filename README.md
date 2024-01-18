# PongAI-QLearning
This is an implementation of the Q-Learning AI agent into a custom pong environment. The agent was able to go undefeated for over 2,500 games to 21 before finally ending the in-game testing for it. Impressively that agent was only trained on 125 episodes (individual 1-point matches). It was only trained on 125 episodes because the final version of training was designed to stop after a certain success percentage (96 percent) inside the batch. This final version of training proved very successful.

The main_QLearning.py File:

Inside this file, you will see the functions that work to make the AI learn and adapt. It also contains the game loop which interacts with the custom pong environment as well as allowing the agent to interact with this environment.

The my_pong_env.py File:

Inside this file, you will see the custom class that creates the custom pong environment that the agent interacts with. This includes all of the backends to making the ball and paddles move, the opponent control, and the implementation of Pygame to be able to visualize the game.
