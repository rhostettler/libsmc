# Sequential Monte Carlo and Markov chain Monte Carlo methods for Matlab
This is a Matlab library implementing sequential Monte Carlo (aka particle filtering and smoothing) as well as particle Markov chain Monte Carlo (PMCMC) methods. The library exclusively makes use of functional programming but makes extensive use of data structures to define models, particle systems, and parameters. The most commonly used structures are:

* `model`: Defines a probabilistic model,
* `p*`: Defines a probability density function,
* `par`: Defines a set of optional parameters for each function,
* `sys`: Contains the particle system (mostly returned from functions, but may also occasionally be used as an input).

Each of the structures is described in turn below, followed by a list of implemented algorithms and standard models included in the toolbox. The interface of each algorithm is documented in the corresponding Matlab help page.

Some general notation used throughout this file (and the code):

* `N`: Number of datapoints,
* `Nx`: State dimension,
* `Ny`: Measurement dimension,
* `M`: Number of samples (particles),
* `K`: Number of MCMC samples.


## Data Structures
### Model Definition
The `model` structure defines a probabilistic model in terms of its state transition density, likelihood, initial state density, and possible static parameters. A model **must** include the following fields:

* `model.px0`: Initial state pdf.
* `model.px`: State transition pdf.
* `model.py`: Likelihood pdf.

Additionally, a model may include the following field:

* `model.ptheta`: **TODO: This is not quite finalized yet.**

The fields `px0`, `px`, and `py` must all be a struct of the probability density type described below. `ptheta` may **TODO: Describe**.

**TODO: Describe the general model here (github doesn't support jax, use a latex generated png instead)**

**TODO: This might be extended at some point to allow for approximate optimal proposals and such**

Certain models may also include additional (specialized) fields (e.g. the conditionally linear models), which are used in algorithms tailored to that class of model. Please see the model section as well as the respective models for more details.


### Probability Densities
Probability density functions (pdfs) need to define a function to draw samples from, a function to evaluate the pdf, a function to evaluate the log-pdf (it is really the log-pdf that is used in most places), and a flag whether these functions can take whole sets of particles at once or not. The fields are:

* `rand(x, t)`: Function to draw samples `z | x` (`z` may be, e.g., the predicted state), 
* `pdf(z, x, t)`: function to evaluate the pdf of `z | x` (`z` may be, e.g., the observation `y`),
* `logpdf(z, x, t)`: function to evaluate the logarithm of the pdf (note that simply implementing `log(pdf())` is a bad idea for numerical reasons), and
* `fast`: A boolean variable which indicates whether whole particle sets can be supplied as arguments or not (to speed up computations).






### The Parameter Structure
TODO: Describe `par`.


### The Particle System
The particle system stores all particles, their weights, ancestor indices, and more, which can be used for debugging or other advanced purposes. The particle system is stored in an array of structs, where each entry corresponds to a time step. Note that there is an additional entry for the initial state (stored in `sys(1)`), that is, if the data length is `N`, then there will be `N+1` entries in `sys`.

The fields of the particle system structure are as follows (not all of them may be present, depending on the method; see the respective method's help function for details):

* `x`: Nx times M matrix of marginal filtering density particles,
* `w`: 1 times M vector of marginal filtering density particle weights,
* `a`: 1 times M vector of ancestor indices,
* `xf`: Nx times M matrix of joint filtering states (i.e. `sys(:).xf(:, j)` corresponds to a complete (degenerate) state trajectory),
* `wf`: 1 times M vector of joint filtering trajectory weights (only set for `sys(N)`), 
* `xs`: Nx times M matrix of smoothed particles, **TODO**
* `ws`: 1 times M vector of smoothed particle weights, **TODO**
* `r`: Boolean variable indicating whether resampling was used at the given time step (for algorithms that use delayed resampling),

`sys` may also contain additional relevant fields, depending on the particular method. See the help page of each function for details (`help <function>`).

**TODO:** I should use individual fields for the marginal density and the joint filtering density; also this needs to be implemented in calculate_particle_lineages (i.e. that they are stored in different fields. need to check how to add a field to arrays of structs).

**TODO:** Make sys an array of structs; that is much more efficient to store and reference things (also with respect to things like covariance matrices.

**TODO: Describe somewhere what the conventions are**

Typical scenarios where we have other variables: s/z/P (for conditionally linear models)

*rb_cpfas:*
* par
  * Nbar  truncation length
* sys
  * x     Nx*M*N, particles of the filter
  * w     1*M*N, weights of the filter
  * P     Nx*Nx*N, covariance matrix of the linear states


*gibbs_pmcmc:*
* par
  * Kburnin
  * Kmixing
  * sample_states
  * sample_parameters

## Algorithms
The library currently implements the following algorithms:

* `sisr_pf`: Generic sequential importance sampling with resampling particle filter.
* `bootstrap_pf`: Bootstrap particle filter.

You might find other methods implemented in the source code, but as long as it is not listed above, consider the implementation to be unstable/subject to major changes/etc.

## Models
There are also a couple of constructors for commonly used model types ready to use. These are found in the folder `src/models` and are prefixed with `model_`. The currently implemented models are:

* `model_lgssm`: Linear, Gaussian state space model,
* `model_wiener_ssm`: Wiener state space model (linear Gaussian dynamics, nonlinear likelihood),
* `model_clgssm1`: Mixing linear/nonlinear Gaussian state space model, (**TODO: Not actually implemented yet**)
* `model_clgssm2`: Hierarchical linear/nonlinear Gaussian state space model. (**TODO: Not actually implemented yet**)


## Examples





## References







## TODO
* Ensure that `par` structures are not passed along to other functions to avoid confusion in what parameters are used and what not.
* The non-Markovian implementations are quite a mess at the moment. We will need to improve that.
* Remove the following function(s):
  * `calculate_incremental_weights()`: This is now being taken care of through the parameter `calculate_incremental_weights()` in `sisr_pf` (and should be migrated to the same approach everywhere else).


