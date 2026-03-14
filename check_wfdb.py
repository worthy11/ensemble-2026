import wfdb
sig, fields = wfdb.rdsamp('train/ecg_train_0001')
print('shape:', sig.shape)
print('leads:', fields['sig_name'])
print('units:', fields['units'])
print('fs:', fields['fs'])
print('sample0:', sig[0])
