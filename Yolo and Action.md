### Abstract

In recent years, significant progress has been made in solving problems using deep reinforcement learning (RL), but in some special environment with multiple units and large action space, such as starcraft2, RL algorithms are difficult or even impossible to achieve good performance.

Our approach can solve such problems in a clever way, through separating  the selection part and the policy part. This combined method allows RL algorithms to be applied to multiple-units environments. We demonstrate our approach's performance on a series of tasks. 

### 1. Introduction

In this paper, we present a new policy architecture which are able to operate in multiple-units environment. 

### 2. Previous Work

### 3. Algorithm303

In this section we describe the detail of 303-algorithm.

##### 3.1. Yolo

In a standard policy gradient approach, the policy is defined by a parameterized function:
$$
\pi: S\rightarrow A
$$
Our method separate this function into two part, the first one called BB net, which is used to choose correct units by bounding box. BB net can be seen as a simplified version of YOLO, for each input state,  it gives a bounding box location. Combined initial state and bounding box location, we will get a new state:
$$
S_{loc} = [S, loc]
$$
In RL framework, we don't have any labels to teach our agent how to bound box correctly under each state, so a initial random strategy is needed, and a Critic will give a score judging the performance of BB net. Obviously the target function of BB net is:
$$
J_{BB}(\theta) = \sum_SV(S_{loc})
$$
3.2. Greedy layer


$$
\pi_BB(s) = argmax_{a \in A}Q(S_{loc}))
$$
