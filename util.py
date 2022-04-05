import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from operator import itemgetter
from tqdm import tqdm


def bin_to_cell(point, grid):
    # lat, lon
    distance = np.linalg.norm(np.array(point)-grid, axis=1)
    nearest = grid[np.where(distance==distance.min())[0]]
    return nearest[0]

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a route."""
    def __init__(self, name='Sampling', **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self._name = name

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.encoder, self.decoder = None, None

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def plot_latent_space(vae, n=10, figsize=20, shape=None, slice2d=[-1,'x',0,'y',1]):
    h, w = (212, 308) if shape is None else shape
    # display a n*n 2D manifold of routes
    scale = 1.0
    figure = np.zeros((h*n, w*n))
    # linearly spaced coordinates corresponding to the 2D plot of route classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    # x, y locations
    x = np.where(np.array(slice2d, dtype='object') == 'x')[0][0]
    y = np.where(np.array(slice2d, dtype='object') == 'y')[0][0]
    slice2d = slice2d.copy()
    for i, yi in tqdm(enumerate(grid_y), file=sys.stdout, total=grid_y.shape[0]):
        for j, xi in enumerate(grid_x):
            slice2d[x], slice2d[y] = xi, yi
            z_sample = np.array([slice2d])
            x_decoded = vae.decoder.predict(z_sample)
            # use mean only
            trace = x_decoded[0].reshape(h, w)
            trace = gaussian_filter(trace, sigma=1)
            figure[
                i * h : (i + 1) * h,
                j * w : (j + 1) * w,
            ] = trace
    plt.figure(figsize=(figsize, figsize))
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    start_range = w // 2
    end_range = n * w + start_range
    pixel_range = np.arange(start_range, end_range, w)
    plt.xticks(pixel_range, sample_range_x, fontsize=18)
    start_range = h // 2
    end_range = n * h + start_range
    pixel_range = np.arange(start_range, end_range, h)
    plt.yticks(pixel_range, sample_range_y, fontsize=18)
    plt.xlabel(f"Dimension-{x+1}", fontsize=24)
    plt.ylabel(f"Dimension-{y+1}", fontsize=24)
    plt.imshow(255-figure, cmap="Greys_r")
    plt.show()
    return

def plot_label_clusters(vae, data, labels, slice2d=[-1,'x',0,'y',1], batch=1000):
    n_batch = np.ceil(data.shape[0]/batch).astype(np.int32)
    x = np.where(np.array(slice2d, dtype='object') == 'x')[0][0]
    y = np.where(np.array(slice2d, dtype='object') == 'y')[0][0]
    # display a 2D plot of the route classes in the latent space
    z_mean = []
    for i in tqdm(range(n_batch), file=sys.stdout, total=n_batch):
        if i+1 == n_batch:
            # zm, _, _ = vae.encoder.predict(data[i:])
            # z_mean.append(zm)
            pass
        else:
            zm, _, _ = vae.encoder.predict(data[i:i+batch])
            z_mean.append(zm)
    z_mean = np.concatenate(z_mean, axis=0)
    plt.figure(figsize=(12,10))
    plt.scatter(z_mean[:,x], z_mean[:,y], c=labels[:z_mean.shape[0]])
    plt.colorbar()
    plt.xlabel(f"z[{x+1}]")
    plt.ylabel(f"z[{y+1}]")
    plt.show()
    return

class FeaturePrediction(tf.keras.Model):
    def __init__(self, input_split, latent_dim, **kwargs):
        super(FeaturePrediction, self).__init__(**kwargs)
        self.n_class, self.n_param = input_split
        self.latent_dim = latent_dim
        self.feature_net = self.build_feature_net()
        self.prediction_net = self.build_prediction_net()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mean_logvar_pred_loss_tracker = tf.keras.metrics.Mean(name="mean_logvar_loss")
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()

    def build_feature_net(self):
        input_class = tf.keras.Input(shape=(self.n_class,1), name='Input_Layer_1')
        input_param = tf.keras.Input(shape=(self.n_param,), name='Input_Layer_2')
        x0 = tf.keras.layers.Conv1D(self.n_param, (self.n_class,), name='Conv1D')(input_class)
        x0 = tf.keras.layers.Flatten(name='Flatten')(x0)
        x = tf.keras.layers.Multiply(name='Multiply')([x0,input_param])
        x = tf.keras.layers.Dense(1024, activation="relu", name='Fully_Connected_1')(x)
        y1 = tf.keras.layers.Dense(256, activation="relu", name='Fully_Connected_2a')(x)
        y2 = tf.keras.layers.Dense(256, activation="relu", name='Fully_Connected_2b')(x)
        mean = tf.keras.layers.Dense(self.latent_dim, name='Mean_Output')(y1)
        logvar = tf.keras.layers.Dense(self.latent_dim, name="LogVar_Output")(y2)
        sample = Sampling(name='Sampling')([mean, logvar])
        feature_extractor = tf.keras.Model([input_class, input_param], [mean, logvar, sample], name="Feature")
        return feature_extractor

    def build_prediction_net(self):
        input_mean = tf.keras.Input(shape=(self.latent_dim,), name='Input_Layer_A')
        input_logvar = tf.keras.Input(shape=(self.latent_dim,), name='Input_Layer_B')
        x1 = tf.keras.layers.Dense(128, activation="relu", name='Fully_Connected_1a')(input_mean)
        x2 = tf.keras.layers.Dense(128, activation="relu", name='Fully_Connected_1b')(input_logvar)
        x = tf.keras.layers.Concatenate(name='Concatenate', axis=1)([x1, x2])
        x = tf.keras.layers.Dense(512, activation="relu", name='Fully_Connected_2')(x)
        x = tf.keras.layers.Dense(64, activation="relu", name='Fully_Connected_3')(x)
        output_layer = tf.keras.layers.Dense(1, name='Output_Layer')(x)
        fuel_predictor = tf.keras.Model([input_mean, input_logvar], output_layer, name="Predictor")
        return fuel_predictor

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.mean_logvar_pred_loss_tracker,
        ]

    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, _ = self.feature_net(X)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            pred_logit = self.prediction_net([z_mean, z_log_var])
            # pred_logit = self.prediction_net(z_sample)
            mean_logvar_pred_loss = self.loss_fn(tf.concat(y, axis=1), tf.concat([z_mean, z_log_var, pred_logit], axis=1))
            # give another weight to the fuel prediction
            mean_logvar_pred_loss += self.loss_fn(y[2], pred_logit)
            total_loss = mean_logvar_pred_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.mean_logvar_pred_loss_tracker.update_state(mean_logvar_pred_loss)
        self.mean_logvar_pred_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mean_logvar_pred_loss": self.mean_logvar_pred_loss_tracker.result(),
        }

class Pathfinder:
    def __init__(self, origin, target, cost, effect=None, diagonal=True, is_custom=True, mode='A*'):
        self.origin = np.array(origin)
        self.target = np.array(target)
        self.cost = cost
        if effect is None:
            effect = np.empty(cost.shape, dtype=np.complex64)
            effect[:] = 0 + 0j
        self.effect = effect
        self.neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], 
                                   [ 0, -1], [ 0, 1], 
                                   [ 1, -1], [ 1, 0], [ 1, 1], ])
        if not diagonal:
            self.neighbors = self.neighbors[[1,3,4,6]]
        self.is_custom = is_custom
        self.mode = mode
        self.height = self.cost.shape[0]
        self.ylim, self.xlim = cost.shape   # row is y, column is x
        self.rotator = lambda mag, theta: np.dot([[np.cos(theta), -np.sin(theta)], 
                                                  [np.sin(theta), np.cos(theta)]], 
                                                 [0, mag]) # counter-clockwise from north
        self.trace = [(np.linalg.norm(self.target-self.origin), np.array([self.origin]))]
        self.is_solvable = True
        self.explored = {tuple(origin)}
        self.path = None

    def get_waypoints(self, current):
        # eight directions from current
        offsets = self.neighbors + current
        # check boundaries
        mask = (offsets[:,0] >= 0) & (offsets[:,1] >= 0)
        mask &= (offsets[:,0] < self.ylim) & (offsets[:,1] < self.xlim)
        valid_offsets = np.array(np.where(mask)).T[:,:2]
        unique_valid_offsets = np.unique(valid_offsets)
        waypoints = self.neighbors[unique_valid_offsets] + current
        # check if not already visited and not wall/no-go-zone
        waypoints = np.array([[x,y] for x, y in waypoints if self.cost[x,y] > 0 and (x,y) not in self.explored])
        return waypoints

    def get_cost(self, rc, current):
        r, c = rc
        cost_g = np.linalg.norm(rc - current)                                       # from current node
        if self.is_custom:
            # customizing pathfinding
            rc_penalty = self.cost[r,c]                                             # penalty by visting undesired cell in cost map
            vec_waypt = rc - current                                                # vector waypoint to convert
            vec_waypt[0], vec_waypt[1] = vec_waypt[1], self.height-1-vec_waypt[0]   # image (row, column) to cartesian (x, y)
            fx_cplx = self.effect[r,c]                                              # get effects coefficient
            vec_effect = self.rotator(fx_cplx.real, fx_cplx.imag)                   # rotate effect magnitude to effect direction
            fx_penalty = np.dot(vec_effect, vec_waypt)                              # factor-in the effect
            cost_g = cost_g * rc_penalty - fx_penalty * fx_cplx.real                # apply to cost_g
        if self.mode == 'A*':
            cost_h = np.linalg.norm(rc - self.target)                               # to target node
            if self.is_custom:
                vec_heuristic = self.target - self.origin
                vec_heuristic[0], vec_heuristic[1] = vec_heuristic[1], self.height-1-vec_heuristic[0]
                fx_penalty_heuristic = np.dot(vec_effect, vec_heuristic)
                cost_h -= fx_penalty_heuristic * fx_cplx.real
            cost = cost_g + cost_h
        else:
            cost = cost_g
        # print(f"calculating: {rc} | rc_penalty {rc_penalty:.3f}, fx_penalty: {fx_penalty:.3f}, cost_h: {cost_h:.3f}, total: {cost:.3f}")
        return cost, rc

    def node_options(self, current):
        waypoints = self.get_waypoints(current)
        # print('adjacent:')
        # print(waypoints)
        adjacent_costs = [self.get_cost(rc, current) for rc in waypoints]
        adjacent_costs = sorted(adjacent_costs, key=itemgetter(0), reverse=False)
        return adjacent_costs
    
    def run(self):
        # main loop
        is_done = False
        while not is_done:
            from_current = []   # container for traces from current node
            for idx, (cost, trace) in enumerate(self.trace[:]):
                # iterate each of progress trace
                node = trace[-1]
                self.explored.add(tuple(node))
                if tuple(node) == tuple(self.target):
                    # solution has been found, break from for loop and main loop
                    is_done = True
                    self.is_solvable = True
                    break
                # print('current node:', node)
                # options from current node, cost and trace
                for cost_new, trace_new in self.node_options(node):
                    cost_trace_new = cost + cost_new, np.append(self.trace[idx][1], [trace_new], axis=0)
                    from_current.append(cost_trace_new)
                # c, t = self.trace[idx]
                # print('costs:', c)
                # print('trace:')
                # print(t)
                # print()
                # add node to the visited node
            if len(from_current) == 0:
                # no more options from current node, break from main loop
                break
            # keep only unique progress nodes with lowest cost
            ctt = np.array(list(map(lambda ct: (ct[0], ct[1][-1][0], ct[1][-1][1]), from_current)))
            keep = []
            for unq in np.unique(ctt[:,1:], axis=0):
                unqrw = np.all(ctt[:,1:]==unq, axis=1)
                minim = ctt[unqrw,0].min()
                lowest = ctt[(unqrw)&(ctt[:,0]==minim)][0]
                lowest_idx = np.where(np.all(ctt==lowest, axis=1))[0][0]
                keep.append(from_current[lowest_idx])
            # replace trace 
            self.trace = sorted(keep, key=itemgetter(0), reverse=False)
        if self.is_solvable:
            self.path = [trace for _, trace in self.trace if tuple(trace[-1]) == tuple(self.target)][0]
        return
