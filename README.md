# Implementation of Popular Swarming Techniques This project implements several of the most popular swarming strategies. ## ReynoldsThis project implements Reynolds Rules of Flocking (or *Boids*), which is based on a balance between three steering forces:- **Cohesion**: the tendency to steer towards the center of mass of neighbouring agents- **Alignment**: the tendency to steer in the same direction as neighbouring agents- **Separation**: the tendency to steer away from neighbouring agents Reynolds (1987) did not provide equations for the steering forces above, so we have taken some artistic liberty and had a bit of fun. In the results section, you will see variations on how "neighbours" are defined. We demonstrate free flocking and flocking to escort (i.e. following a reference). A more formal definition and analysis of flocking was provided by Olfati-Saber (2006), which we have implemented [here](https://github.com/tjards/flocking_network)Craig Reynolds, ["Flocks, Herds, and Schools:A Distributed Behavioral Model"](https://www.red3d.com/cwr/papers/1987/boids.html), *Computer Graphics, 21(4) (SIGGRAPH '87 Conference Proceedings)*, pages 25-34, 1987.<p float="center">  <img src="https://github.com/tjards/swarming_sim/blob/master/Figs/animation_reynolds_01.gif" width="45%" /></p>## SaberThis project implements flocking for a large network of aerial vehicles. The strategy is based on the (elegant) 3-part distributed formulation proposed in:Reza Olfati-Saber, ["Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory"](https://ieeexplore.ieee.org/document/1605401), *IEEE Transactions on Automatic Control*, Vol. 51 (3), 2006.The formulation involves three distinct terms:1. **Alpha-Alpha Interaction (alignment)** - encourages the agents to align in a lattice formation. The function represents other vehicles as "alpha" agents. 2. **Alpha-Beta Interaction (obstacle avoidance)** - discourages the agents from colliding with obstacles. The function represents obstacles as "beta" agents.3. **Gamma Feedback (navigation)** - encourages motion towards a static or dynamic target. The function represents the target as a "gamma" agent.When combined together, these terms produce an emergent behaviour that is refered to as "flocking". <p float="center">    <img src="https://github.com/tjards/swarming_sim/blob/master/Figs/animation_saber_01.gif" width="45%" /></p>## Starling FlockingH. Hildenbrandt, C. Carere, and C.K. Hemelrijk,["Self-organized aerial displays of thousands of starlings: a model"](https://academic.oup.com/beheco/article/21/6/1349/333856?login=false), *Behavioral Ecology*, Volume 21, Issue 6, pages 1349–1359, 2010.<p float="center">  <img src="https://github.com/tjards/swarming_sim/blob/master/Figs/animation_starling.gif" width="45%" /></p>## EncirclementThis work is related to the following research in multi-agent robotics:Ahmed T. Hafez, Anthony J. Marasco, Sidney N. Givigi, Mohamad Iskandarani, Shahram Yousefi, and Camille Alain Rabbath, ["Solving Multi-UAV Dynamic Encirclement via Model Predictive Control"](https://ieeexplore.ieee.org/document/7066874), *IEEE Transactions on Control Systems Technology*, Vol. 23 (6), Nov 2015<p float="center">  <img src="https://github.com/tjards/swarming_sim/blob/master/Figs/animation_circle_01.gif" width="45%" /></p>## Dynamic Lemniscateto be added<p float="center">    <img src="https://github.com/tjards/swarming_sim/blob/master/Figs/animation_lemni_01.gif" width="45%" /></p># CitingThe code is opensource but, if you reference this work in your own reserach, please cite me. I have provided an example bibtex citation below:`@techreport{Jardine-2022,  title={Swarming Simulator},  author={Jardine, P.T.},  year={2022},  institution={Royal Military College of Canada, Kingston, Ontario},  type={Technical Report},}`Alternatively, you can cite any of my related papers, which are listed in [Google Scholar](https://scholar.google.com/citations?hl=en&user=RGlv4ZUAAAAJ&view_op=list_works&sortby=pubdate).# Some plots 