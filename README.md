# Poisson process models

These are models for Poisson processes.  Goal is to be able
to forecast values in a Poisson process that has cyclic
fluctuation in the underlying process.  This means that there 
is an underlying variability to the Poisson process, but the 
variability is cyclic.  Another way of describing that would 
be there is seasonality to the underlying Poisson process.  An
example would be the hourly rate of traffic. The process is 
Poisson, but the underlying process is parameterized as a 
function of time. In the middle of the night the process is 
paramaterized differently than during rush hour. This is assumed
to be cyclic since it would have the same pattern every day.

Since we want to actually track the data, there are some 
good models for stochastic systems that have to be discarded
since they do not attempt to track the variability, but rather
attempt to estimate the mean and variance.  Techniques that 
fall into this camp include GARCH and Kalman filters.  This
basically leaves us with ARIMA models and the use of a generic
model such as LSTM from deep learning, both of which will attempt
to forecast the next actual data point.

This experiment entails a cyclic underlying Poisson process that
changes.  Reason for this is, in the case of the traffic 
example, there is a difference between weekday and weekend 
traffic.  The goal, of course, is that the model we use will be
able to remain relevant when it is exposed to underlying 
patterns in the process that it has never seen.  To judge
the fitness of the model, we use the RMSE on a test set and
qualitatively observe the residual.  The idea being that we 
want the RMSE to be as small as possible and the residual
to appear random, or at least lack evidence of the underlying
process.

Since there is seasonality to the underlying model, the 
assumption is that in all training it would be prudent to 
train on at least one full season.  In practice this seems
to work better with at least 2 seasons, and for the notebooks
presented they were trained with 3 seasons.  The different types
of model were trained differently.  The LSTM model was trained 
with an abundance of pre-generated model data, whereas the 
ARIMA model was trained on the actual test data as a rolling
model with the most recent point being the data point immediately
preceding the comparison point. To make the plots comparable, you
will see that some of the prior data is truncated.

The LSTM model presents some challenges. First, with a random
process, we do not have a priori knowledge of the scale.  If we
make assumptions and shrink the data too much it tends to not
train as well.  On the other hand, if the test data is larger 
than the training set it is impossible for the model to 
replicate the large values. Furthermore, the model does not 
generalize very well across different underlying processes.  It
qualitatively captures the essence of the trend, but the 
RMSE is not good.  The residual tends to prominently show the
underlying Poisson control.  It is also very expensive to 
train the LSTM.  This forces the idea that the model would
need to be trained in advance with knowledge of expected process
variability.  This may not always be possible.  Since the model
does not seem to generalize well, it does not seem an ideal
candidate for this specific use.

The ARIMA model is implemented as a rolling model of the preceding
few seasons.  While this is somewhat expensive, it is significantly
faster than training the LSTM model (seconds versus hours).  There is
an advantage to the ARIMA model in that it continuously tracks the 
pattern since it is a rolling pattern model.  This seems to perform
better than the training all in advance idea that is central to the
LSTM model. In theory, so long as the ARIMA model remains compatible
with the underlying data, it should retain similar power in fitting
and forecasting the new data.  Some problems encountered include
convergence issues with higher order ARIMA models and a lack of 
ability to get good RMSE performance.  The residual always retains
some of the underlying Poisson control process despite using 
differentiated data to improve stationarity and seasonality 
performance.  If you change to a simpler underlying model using a
Gaussian process, both the RMSE and the quality of the 
residual improve dramatically.  This makes me think that the largest
problem with the ARIMA model on this data is general incompatibility
with the type of data, fluctuating Poisson data.  

Between the LSTM and ARIMA model, the ARIMA model performs better,
requires significantly less computation time to achieve a result,
is much simpler to tune, and is less susceptible to data 
variability.  There may be ways to improve the performance. The specific
way that the data is generated here may not be the most realistic.  It
is step discontinuous rather than smooth with regard to the underlying
Poisson characteristic.  While the process itself appears to be 
Poisson, perhaps in practice it is adequate to suppose that it is 
actually Gaussian.  If this hypothesis is true, it is reasonable
to anticipate better results.  Of course the only way to know is 
to collect real data and repeat the experiment.

Even by normalizing the Poisson distribution by the mean doesn't
affect the quality of the residual.  The seasonal undulations are
removed, but the variability was nearly identical in terms of the
qualitative appearance of a qq-plot.  To be fair, this is expected since
the variance of the distribution was not normalized, only the mean.

Another mechanism for improvement might be adding time series 
components to the ARIMA model.  Particularly for constant underlying
changes in the Poisson process, these could be modeled by a 
Fourier series.  Despite the underlying process being assumed to
change, the Fourier terms might help normalize the data and remove
some of the artifacts from the residual.

