import numpy as np

def get_channel_gain(self, f_hz, ptx_dbm):

		def db_to_mw(db):
			return 10.**(db/10.)

		def mw_to_db(mw):
			return 10.*np.log10(mw)

		# First, measure just the noise level on the receiving node.
		# Transmitter is turned off.
		pnoise_dbm = self.measure(f_hz, None)

		# Second, measure the received signal power level with the
		# transmitter turned on.
		prx_dbm = self.measure(f_hz, ptx_dbm)

		# Convert all power values to linear scale.
		ptx_mw = db_to_mw(ptx_dbm)
		pnoise_mw = db_to_mw(pnoise_dbm)
		prx_mw = db_to_mw(prx_dbm)

		# Take the mean of both noise and received signal power
		# measurements.
		pnoise_mw_mean = np.mean(pnoise_mw)
		prx_mw_mean = np.mean(prx_mw)

		#print "p_noise = %.1f dBm (mean=%e mW std=%e mW)" % (
		#		mw_to_db(pnoise_mw_mean),
		#		pnoise_mw_mean,
		#		np.std(pnoise_mw))
		#print "p_rx    = %.1f dBm (mean=%e mW std=%e mW)" % (
		#		mw_to_db(prx_mw_mean),
		#		prx_mw_mean,
		#		np.std(prx_mw))

		# Use the mean values to estimate the channel gain.
		h_mean = (prx_mw_mean - pnoise_mw_mean)/ptx_mw

		# Convert back to logarithmic scale.
		h_mean_db = mw_to_db(h_mean)

		#print "h = %.1f dB" % (h_mean_db,)

		return h_mean_db



c_gain = get_channel_gain(100,10)

print(c_gain)
