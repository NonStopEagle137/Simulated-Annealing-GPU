import tensorflow.compat.v1 as tf
import numpy as np

class gpu_simulated_annealing:
    def __init__(self, init_temp,n_dims,iter__,decay_rate, learning_rate, cooling_rate):
        #self.config = tf.ConfigProto(device_count = {'GPU': 1})
        self.iter__ = iter__
        self.n_dims = n_dims
        """Initializing Tensors."""
        self.x_guess = self.get_x_guess()
        self.present_temperature = tf.Variable(init_temp, dtype = tf.float64)
        self.neg_one = tf.constant(-1, dtype = tf.float64)
        self.cooling_rate = tf.constant(cooling_rate, dtype = tf.float64)
        self.decay_rate = tf.constant(decay_rate, dtype = tf.float64)
        self.prob = tf.Variable(0.43, dtype = tf.float64) # random values
        self.E_new = tf.Variable(0.1, dtype = tf.float64)
        self.E_n = tf.Variable(0.1, dtype = tf.float64)
        self.ret_val = tf.Variable([0,1], dtype = tf.float64)
        self.threshold = tf.constant(0.5, dtype = tf.float64)
        self.learning_rate = tf.constant(learning_rate, dtype = tf.float64)
        self.x_new = self.get_x_new()
        """ Variables are initialised."""
    def get_x_guess(self):
        x = tf.Variable(tf.truncated_normal((self.n_dims,), dtype = tf.float64))
        return x
    def get_x_new(self):
        x = tf.Variable(tf.truncated_normal((self.n_dims,), dtype = tf.float64))
        return x
    def cost_function_old(self):
        #fun = tf.math.sin(vector**7) + tf.math.cos(vector**3)
        power = tf.multiply(tf.ones(shape = self.x_guess.shape, dtype = tf.float64), 4.0)
        fun = tf.math.reduce_sum(tf.math.pow(self.x_guess, power))
        self.E_n = self.E_n.assign(fun)
        
    def cost_function_new(self):
        #fun = tf.math.sin(vector**7) + tf.math.cos(vector**3)
        power = tf.multiply(tf.ones(shape = self.x_new.shape, dtype = tf.float64), 4.0)
        fun = tf.math.reduce_sum(tf.math.pow(self.x_new, power))
        self.E_new = self.E_new.assign(fun)
    def eval_true(self):
            self.ret_val = self.ret_val.assign([0,1])  
    def eval_false(self):
            self.ret_val = self.ret_val.assign([1,0])
    def prob_alt(self):
            self.prob = self.prob.assign(tf.math.exp(tf.multiply(self.neg_one,
                                       tf.divide(tf.subtract(self.E_new,self.E_n),
                                                 tf.multiply(self.present_temperature,
                                                             self.cooling_rate)))))
            if self.prob.read_value() > self.threshold:
                self.eval_true()
            else:
                self.eval_false()
    def prob__(self):
        if self.E_n >= self.E_new:
            self.eval_true()
        else:
            self.prob_alt()
        

    def anneal(self):
        for i in range(self.iter__):
            self.present_temperature = self.present_temperature.assign(tf.multiply(\
                                                                 self.present_temperature,
                                                                 self.decay_rate))
            
            self.cost_function_old()
                
            self.x_new = self.x_new.assign(tf.random.uniform(minval = -1.0*self.learning_rate,
                                                      maxval = 1.0*self.learning_rate,
                                                      shape = self.x_new.shape,
                                                      dtype = tf.float64))
            self.cost_function_new()
            self.prob__() 
            if self.ret_val[0] == 0:
                self.x_guess = self.x_guess.assign(self.x_new.read_value())
                print("[INFO] Current Cost : ", self.E_new.numpy())
            else:
                print("[INFO] Current Cost : ", self.E_n.numpy())
                

init_temp = 5000
n_dims = 5000000
iter__ = 15
decay_rate = 0.9
learning_rate = 0.01
cooling_rate = 0.95
anneal_obj = gpu_simulated_annealing(init_temp,n_dims,iter__,decay_rate, learning_rate, cooling_rate)
anneal_obj.anneal()