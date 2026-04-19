# Predictive driving 

For a self driving car (ego) to make intelligent decisions, the ability to know how 
other agents (ado) move in the scene is essential. The traditional way of incorporating agent predictions has been either use them as an input to the planner or just treat them as an auxiliary task with the aim of building a robust scene representation. 

This project explores the idea of learning a robust autonomous driving policy that can detect and react to adversarial or Out-of-Distribution (OOD) agents. We aim to achieve this by maintaining a generative prior of "nominal" driving behavior and computing an anomaly score for observed agents using a learned discriminator. 

### Videos

- [Highway scenario video](./assets/highway_1.mp4)
- [Roundabout scenario video](./assets/roundabout_1.mp4)
