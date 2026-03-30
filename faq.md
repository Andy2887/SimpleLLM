Q: Explain your project development process.

A: 

Read Llama paper and Meta's official example code.

1. Unit testing with fixed input, output and parameter values.

2. Save metrics and analyze training loss trends.

Q: What challenges did you encounter during building this project?

A: 

First, how to make sure the weights loads into my model.

1. Check the weight keys and shapes in Meta's checkpoint.
2. Create a function to map Meta's key to the names of my key.
3. load my model's dict and diff the keys.
4. If no difference, check the shape.