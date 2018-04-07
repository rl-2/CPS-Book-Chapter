import numpy as np
import torch
from scipy import misc
import time
import argparse
import ipdb
import drone
from pytorchrl import Agent
import utilities as util
import pickle

parser = argparse.ArgumentParser(description='RL for block collection environment.')
parser.add_argument('--mode', type=str, choices=['eval', 'imitation', 'rl'], default='eval')
#parser.add_argument('--parameter_path', type=str, default='Selected_models/poor_performance.pkl')
#parser.add_argument('--parameter_path', type=str, default='Selected_models/only_get_boxes_on_right.pkl')
parser.add_argument('--parameter_path', type=str, default='Selected_models/high_performance.pkl')
parser.add_argument('--init_from_model', type=bool, default=True) # false -> start from fresh, true -> build on the exsiting model 
parser.add_argument('--greedy', type=bool, default=False)
parser.add_argument('--model', type=str, default='CNN')
parser.add_argument('--discount_factor', type=float, default='0.99')
parser.add_argument('--learning_rate', type=float, default='.0001') # .0001 works well as default for 3 boxes
parser.add_argument('--scheduler_step_size', type=int, default='5')
parser.add_argument('--scheduler_gamma', type=float, default='.99')
parser.add_argument('--num_episodes', type=int, default=30000)
parser.add_argument('--max_steps', type=int, default=300) 
parser.add_argument('--num_actions', type=int, default=3)
parser.add_argument('--save_visual', type=bool, default=True)
parser.add_argument('--logfile', type=str, default='logfile.txt')
parser.add_argument('--image_size', type=str, choices=['small', 'large'], default='large')
parser.add_argument('--model_save_rate', type=int, default='250')

# added by Rodger 
parser.add_argument('--box_num', type=int, default=3)
parser.add_argument('--advantage_type', type=int, default=4) # See pytorchrl.py for behavior

args = parser.parse_args()

logfile = open(args.logfile, 'w+')

# Print arguments
print("Passed arguments")
for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

def run_episode(env, agent, episode):
    done = False
    S = [] # state
    A = [] # action
    R = [] # reward
    P = [] # probability 

    box_collected = 0
 
    s = env.reset()
 
    if args.mode == "imitation":
        action_ready = False
        a = 0
        prob = 0
    elif args.mode == "rl" or args.mode == "eval":
        action_ready = True
        a = 0
        prob = torch.autograd.Variable(torch.Tensor([[1.0]]))

    step = 0

    while not done:
        if args.mode == "rl" or args.mode == "eval":
            if args.model == "CNN":
                s_temp = s
                s = util.processImage(s)

            #TODO(Sam): Is the following copy because of a bug?
            last_a, last_prob = a, prob
            a, prob = agent.select_action(s, greedy=args.greedy)

            
            #ipdb.set_trace()

            if args.save_visual is True:
                # base_filename = "visuals/{:03d}_{:03d}_{:02d}_{:.2f}".format(episode, step, last_a, last_prob.data[0,last_a])
                base_filename = "tSNE_images/good_model/{:03d}_{:03d}_{:02d}".format(episode, step, a)
                img_filename = base_filename+".png"
                pkl_filename = base_filename+".pkl"
                print("Saving {} .png and .pkl.".format(base_filename))
                misc.imsave(img_filename, s_temp)                
                #file = open(pkl_filename, "wb")
                #pickle.dump(s, file)
                # file.close()
                #ipdb.set_trace()

        x = util.kbfunc()
        
        if action_ready is False:
            if x != False and x.decode() == 'w':
                a = 0
                action_ready = True
            elif x != False and x.decode() == 'a':
                a = 1
                action_ready = True
            elif x != False and x.decode() == 'd':
                a = 2
                action_ready = True
            elif (x != False and x.decode() == 'f'):
                done = True
            elif (x != False and x.decode() == 'q'):
                return (True, [0])


        elif action_ready is True:
            # hack for storing the processed image during imitation learning
            if args.mode == "imitation":
                s = util.processImage(s)

            s2, r, done, box_collected = env.step(a)

            S.append(s)
            A.append(a)
            R.append(r)
            P.append(prob)
     
            s = s2

            if args.mode == "imitation":
                action_ready = False

        if done:
            # todo: Only update NN when box was hit--everything else has 0 reward.
            # --penalty-- if max(R) > 0 and len(R) > 1:
            if len(R) > 1: 
                logfile.write(str(sum(R)))
                logfile.flush()

                if args.mode == "imitation" or args.mode == "rl":
                    success = agent.fit(S, A, R, args.advantage_type) # success is for handling a glitch where boxes pop up under the drone
                    # todo: hack for handling a glitch where boxes pop up under the drone
                    if not success:
                        R = [0]

            else:
                logfile.write("0")
                logfile.flush()                

            env.reset()
            time.sleep(1)

        step += 1

    # max(R) being returned to handle a glitch
    return(False, R, box_collected)

   
if __name__ == '__main__':
    # Will end episode after args.max_steps
    # env = drone.environment(max_steps = args.max_steps, image_size=args.image_size)
    env = drone.environment_2(max_steps = args.max_steps, image_size=args.image_size, box_num=args.box_num)

    # Used for setting up the MLP. Ignored for CNN.
    num_pixels = len(env.getImage().flatten())
    agent = Agent(args.model, num_pixels, args.num_actions, args.max_steps, args.discount_factor, args.learning_rate, args.scheduler_step_size, args.scheduler_gamma)

    # Used for tracking rewards over time
    running_reward = 0

    if args.mode == "imitation":

        for episode in range(1, args.num_episodes):
            print("Episode {}".format(episode))
            quit, reward, box_collected = run_episode(env, agent, episode)

            print("Saving DNN parameters to {}".format(args.parameter_path))
            torch.save(agent.policy, args.parameter_path)        
            
            if quit:
                break

        # Bad style, see above for loop
        if not quit:
            print("Saving DNN parameters to {}".format(args.parameter_path))
            torch.save(agent.policy, args.parameter_path)

    elif args.mode == "rl":
        if args.init_from_model:
            print("Loading model parameters from {}".format(args.parameter_path))
            agent.policy = torch.load(args.parameter_path)

        for episode in range(1, args.num_episodes):
            print("Episode {}".format(episode))
            quit, reward, box_collected = run_episode(env, agent, episode)        
            if quit:
                break

            running_reward = .9*running_reward + .1*sum(reward)
            print("running reward: {}".format(running_reward))

            if running_reward > 2.99:
                print("Saving DNN parameters to {}".format(args.parameter_path))
                torch.save(agent.policy, "highscore_models/model-high-avg-" + str(episode) + ".pkl") 

            if episode % args.model_save_rate == 0:
                torch.save(agent.policy, args.parameter_path)


            # if len(reward) > 1:
            #     print("Saving DNN parameters to {}".format(args.parameter_path))
            #     torch.save(agent.policy, args.parameter_path)

            # if running_reward > 2.75:
            #     print("Saving DNN parameters to {}".format(args.parameter_path))
            #     torch.save(agent.policy, "highscore_models/model-high-avg-" + str(episode) + ".pkl") 

            # if box_collected > 2:
            #     torch.save(agent.policy, "highscore_models/model-" + str(episode) + ".pkl") 

    elif args.mode == "eval":
        print("Loading model parameters from {}".format(args.parameter_path))
        agent.policy = torch.load(args.parameter_path)
        agent.policy.eval()

        for episode in range(1, args.num_episodes):
            print("Episode {}".format(episode))
            quit, reward, box_collected = run_episode(env, agent, episode)        
            if quit:
                break

    logfile.close()