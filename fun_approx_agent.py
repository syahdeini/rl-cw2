import cv2
import numpy as np
import pdb
from enduro.agent import Agent
from enduro.action import Action


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # The horizon defines how far the agent can see
        self.horizon_row = 5
        self.grid_cols = 10
        self.num_features = 5

        # The state is defined as a tuple of the agent's x position and the
        # x position of the closest opponent which is lower than the horizon,
        # if any is present. There are four actions and so the Q(s, a) table
        # has size of 10 * (10 + 1) * 4 = 440.
        self.Q = np.ones((self.grid_cols, self.grid_cols + 1, 4))

        # Add initial bias toward moving forward. This is not necessary,
        # however it speeds up learning significantly, since the game does
        # not provide negative reward if no cars have been passed by.
        self.Q[:, :, 0] += 1.

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}

        # Learning rate
        self.alpha = 0.01
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01

        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []

        # initialize parameter with normal value
        mu, sigma = 0, 0.1 # mean and standard deviation
        self.features =   np.zeros((len(self.getActionsSet()),self.num_features))
        # initialize number of features
        self.parameters  =  np.random.normal(mu, sigma,(1,self.num_features))
        self.next_features = self.features   



    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        # Reset the total reward for the episode
        self.total_reward = 0
        self.next_state = self.buildState(grid)
        
    
    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal


        self.features = self.next_features
        # decide which action to take
        # If exploring
        Q_s = np.dot(self.parameters,self.features.T)[0]
        if np.random.uniform(0., 1.) < self.epsilon:
            probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
            idx = np.random.choice(4, p=probs)
            self.action = self.idx2act[idx]
            print("explore")
        else:
            # Select the greedy action
            self.action = self.idx2act[np.argmax(Q_s)]
            print("features",self.features)
            print("parameter",self.parameters)

            print("greedy")
            print('----',Q_s)
            # cv2.waitKey(200)

        self.reward = self.move(self.action)
        self.Q_s = np.dot(self.parameters, self.features[self.act2idx[self.action]].T)[0] # scalar
        self.total_reward += self.reward

        print("ACTION",Action.toString(self.action))
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work
        pass

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        # sense the next state to update s+1
        # self.next_state = self.buildState(grid)
        self.calculateFeatures(road,cars,speed,grid)
        self.next_features = self.meta_features


    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        # update parameter
        next_Q_s  = np.dot(self.parameters,self.next_features.T)
        next_Q_s = np.max(next_Q_s) # scalar
        _feature = self.features[self.act2idx[self.action]]
        self.parameters = np.add(self.parameters,self.alpha * (self.reward + self.gamma*(next_Q_s-self.Q_s))* _feature)
        # print(self.parameters)


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # You could comment this out in order to speed up iterations
        cv2.imshow("Enduro", self._image)
        cv2.waitKey(40)

    ### FEATURE ###########################
    def set_all_feature(self, idx,val):
        for act in self.getActionsSet():
            self.meta_features[self.act2idx[act]][idx]=val
        
    def feature1(self,grid,pos_player):
         # check if car infront of player
        front_car=grid[:,pos_player]
        # ignore player
        front_car[0] -= 2
        front_car = np.sort(np.argwhere(front_car>0).flatten())
        # check jarak enemy infront of car
        if len(front_car)!=0 and front_car[0] < 2:
           self.meta_features[self.act2idx[Action.BRAKE]][0] = -0.5
           self.meta_features[self.act2idx[Action.LEFT]][0] = 1
           self.meta_features[self.act2idx[Action.RIGHT]][0] = 1
        else:
           self.meta_features[self.act2idx[Action.ACCELERATE]][0] = 1
           self.meta_features[self.act2idx[Action.BRAKE]][0]  = 0

         
    def feature2(self,pos_player):
        # check if it near the left wall or right wall and there is an enemy in-front of the player
        self.set_all_feature(1,0)
        self.meta_features[self.act2idx[Action.ACCELERATE]][1] = 1            
        if pos_player < 2: #it's too left
           self.meta_features[self.act2idx[Action.RIGHT]][1] = 1            
           self.meta_features[self.act2idx[Action.LEFT]][1] = 0            
        elif pos_player > 8:
           self.meta_features[self.act2idx[Action.RIGHT]][1] = 0            
           self.meta_features[self.act2idx[Action.LEFT]][1] = 1          

    # if collision happen speed will be negative we encourage right and left
    def feature3(self,cars):
        # if cars['others']>0:
        #     for idx,enemy in enumerate(cars['others']):
        #         x,y,w,h = enemy
        #         cars['others'][idx]=(x-1)
        if self.collision(cars):
            self.meta_features[self.act2idx[Action.RIGHT]][2] = 1            
            self.meta_features[self.act2idx[Action.LEFT]][2] = 1

    # collision already happen
    def feature4(self,speed):
        if speed < 0:
            self.meta_features[self.act2idx[Action.ACCELERATE]][3] = 1           
    # def feature4(self,spee)

    def feature5(self,pos_player,grid):
        # pdb.set_trace()
        front = grid[:,pos_player]
        front[0] -= 2
        _front = np.sort(np.argwhere(front > 0).flatten())
        _front = _front[0] if len(_front)>0 else None
        left_player = grid[:,pos_player+1]
        _left = np.sort(np.argwhere(left_player > 0).flatten())
        _left = _left[0] if len(_left)>0 else None
        
        right_player = grid[:,pos_player+1]
        _right = np.sort(np.argwhere(right_player > 0).flatten())
        _right = _right[0] if len(_right)>0 else None

        if _left and _right and _front:
            if _left<3 and _right<3 and _front <3:
                self.meta_features[self.act2idx[Action.BRAKE]][4] = 1
                self.meta_features[self.act2idx[Action.RIGHT]][4] = 0.5            
                self.meta_features[self.act2idx[Action.LEFT]][4] = 0.5
            elif _left<3 and _right<3:
                self.meta_features[self.act2idx[Action.ACCELERATE]][4] = 1
            elif _right<3:
                self.meta_features[self.act2idx[Action.LEFT]][4] = 1
            elif _left<3:
                self.meta_features[self.act2idx[Action.RIGHT]][4]=1
    def calculateFeatures(self,road,cars,speed,grid):
        # features
        # feature 1. if there is no car infront of player (min range:3)
        ori_grid = np.array(grid)
        self.meta_features =   np.zeros((len(self.getActionsSet()),self.num_features))
        [[x]] = np.argwhere(grid[0,:] == 2)
        pos_player = x
        

        # 0 will sum rows
        pos_column = np.sum(grid,axis=0) # column [liat kedepan]
        pos_rows = np.sum(grid,axis=1) # liat ke samping

        # ignore the agent
        pos_rows[0] -= 2
        pos_column[pos_player] -= 2
        # rows = np.sort(np.argwhere(pos_column > 0).flatten())
        # pos_col_enemy = rows[0] if len(rows)>0 else -1
        # col = np.sort(np.argwhere(pos_rows > 0).flatten())
        # pos_row_enemy = rows[0] if len(col)>0 else -1 
        self.feature1(grid,pos_player)
        self.feature2(pos_player)
        self.feature3(cars)
        self.feature4(speed)
        self.feature5(pos_player,ori_grid)
        print("speed",speed)
        print("pos player",pos_player)
    ### END OF FEATURE ###################
    def buildState(self, grid):
        state = [0, 0]

        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        state[0] = x

        # Sum the rows of the grid
        rows = np.sum(grid, axis=1)
        # Ignore the agent
        rows[0] -= 2
        # Get the closest row where an opponent is present
        rows = np.sort(np.argwhere(rows > 0).flatten())

        # If any opponent is present
        if rows.size > 0:
            # Add the x position of the first opponent on the closest row
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:
                    # 0 means that no agent is present and so
                    # the index is offset by 1
                    state[1] = i + 1
                    break
        return state

    def choose_action(self):
        #np.dot(self.parameters.T,self.features)
        #using argmax select which action to take
        pass
if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
