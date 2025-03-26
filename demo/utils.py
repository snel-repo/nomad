import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from lfads_tf2.tuples import LFADSInput

def chop_and_infer(func,
                   data,
                   out_fields,
                   seq_len=30,
                   stride=1,
                   batch_size=64,
                   output_dim=None):
    """
    Chop data into sequences, run inference on those sequences, and merge
    the inferred result back into an array of continuous data. When merging
    overlapping sequences, only the non-overlapping sections of the incoming
    sequence are used.
    Parameters
    ----------
    func : callable
        Function to be used for inference. Must be of the form:
            prediction = func(data)
    data : `numpy.ndarray` of shape (n_samples, n_features)
        Data to be split up into sequences and passed into the model
    seq_len : int, optional
        Length of each sequence, by default 30
    stride : int, optional
        Step size (in samples) when shifting the sequence window, by default 1
    batch_size : int, optional
        Number of sequences to include in each batch, by default 64
    output_dim : int, optional
        Number of output features from func, by default None. If None,
        `output_dim` is set equal to the number of features in the input data.
    Returns
    -------
    output : numpy.ndarray of shape (n_samples, n_features)
        Inferred output from func
    Raises
    ------
    ValueError
        If `stride` is greater than `seq_len`
    """
    if stride > seq_len:
        raise ValueError(
            "Stride must be less then or equal to the sequence length")

    data_len, data_dim = data.shape[0], data.shape[1]
    output_dim = {k: data_dim for k in out_fields} if output_dim is None else output_dim

    batch = np.zeros((batch_size, seq_len, data_dim), dtype=data.dtype)
    output = {k: np.zeros((data_len, output_dim[k]), dtype=data.dtype) for k in out_fields}
    olap = seq_len - stride

    n_seqs = (data_len - seq_len) // stride + 1
    n_batches = np.ceil(n_seqs / batch_size).astype(int)

    i_seq = 0  # index of the current sequence
    for i_batch in range(n_batches):
        n_seqs_batch = 0  # number of sequences in this batch
        # chop
        start_ind_batch = i_seq * stride
        for i_seq_in_batch in range(batch_size):
            if i_seq < n_seqs:
                start_ind = i_seq * stride
                batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
                                                   seq_len]
                i_seq += 1
                n_seqs_batch += 1
        end_ind_batch = start_ind + seq_len
        # infer
        # batch_out = func(batch)[:n_seqs_batch]
        batch_out = {k:v[:n_seqs_batch] for k,v in func(batch).items()}
        n_samples = n_seqs_batch * stride

        # merge
        if start_ind_batch == 0:  # fill in the start of the sequence
             for k in out_fields: 
                output[k][:olap, :] = batch_out[k][0, :olap, :]

        out_idx_start = start_ind_batch + olap
        out_idx_end = end_ind_batch
        out_slice = np.s_[out_idx_start:out_idx_end]
        for k in out_fields: 
            output[k][out_slice, :] = batch_out[k][:, olap:, :].reshape(
                n_samples, output_dim[k])

    return output


def get_causal_model_output(model, 
                            binsize, 
                            input_data, 
                            out_fields, 
                            output_dim, 
                            stride=1, 
                            batch_size=64): 
    
    '''
    Get the output of the model for the input data, causally.
    
    Parameters
    ----------
    model : LFADS object
        The trained LFADS model
    binsize : float
        The binsize of the input data in seconds
    input_data : np.ndarray
        The input data to be used for inference, of shape time x channels
    out_fields : list
        The fields of the model output to return
    output_dim : dict
        Dictionary where each key is an entry in out_fields and each value is
        the corresponding dimension of the output data
    stride : int, optional
        The stride of the model, by default 1
    batch_size : int, optional
        The batch size of the model, by default 64

    Returns
    -------
    model_output : dict
        The output of the model for the input data, where each key is an 
        entry in out_fields and each value is the corresponding output data
    '''

    # make sure input data is appropriate level of precision 
    input_data = input_data.astype('float64')

    # pass means instead of samples 
    model.cfg['MODEL']['SAMPLE_POSTERIORS'] = False 
    
    seq_len = model.cfg.MODEL.SEQ_LEN
    data_dim = model.cfg.MODEL.DATA_DIM

    rng = np.random.default_rng()
    data = np.zeros((batch_size, seq_len, data_dim), dtype=np.float32)
    ext_input_dim = model.cfg.MODEL.EXT_INPUT_DIM
    ext_input = np.zeros((batch_size, seq_len, ext_input_dim), dtype=np.float32)
    dataset_name = np.full(shape=[1], fill_value='')
    behavior = np.zeros((batch_size, seq_len, 0), dtype=np.float32)

    # Chop the data into sequences and run inference
    # how much to shift the window for each sequence
    seq_len_non_fp = model.cfg.MODEL.SEQ_LEN

    def lfads_infer(data_in):            
        data[:, :seq_len_non_fp, :] = data_in # pad the data with zeros in the forward prediction bins
        # run inference
        lfads_input = LFADSInput(enc_input=data,
                                ext_input=ext_input,
                                dataset_name=dataset_name,
                                behavior=behavior)
        lfads_output = model.graph_call(lfads_input)

        out_dict = {}
        for out_field in out_fields:
            out_dict[out_field] = getattr(lfads_output, out_field)[:, :seq_len_non_fp, :].numpy()
        # leave out the forward prediction bins
        return out_dict

    model_output = chop_and_infer(
        lfads_infer,
        input_data,
        out_fields,
        stride=stride,
        seq_len=seq_len_non_fp,
        batch_size=batch_size,
        output_dim=output_dim
    )

    return model_output


def generate_lagged_matrix(input_matrix: np.ndarray, lag: int):
    """
    Generate a lagged version of an input matrix.

    Parameters:
    input_matrix (np.ndarray): The input matrix.
    lag (int): The number of lags to consider.

    Returns:
    np.ndarray: The lagged matrix.
    """
    # Initialize the lagged matrix
    lagged_matrix = np.zeros((input_matrix.shape[0] - lag, input_matrix.shape[1] * (lag + 1)))

    # Fill the lagged matrix
    for i in range(lag + 1):
        lagged_matrix[:, i*input_matrix.shape[1]:(i+1)*input_matrix.shape[1]] = input_matrix[lag-i : (-i if i != 0 else None)]

    return lagged_matrix


def fit_and_eval_decoder(
    train_rates: np.ndarray,
    train_behavior: np.ndarray,
    eval_rates: np.ndarray,
    eval_behavior: np.ndarray,
    grid_search: bool=True,
    return_preds: bool=False
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates :
        2d array time x units.
    train_behavior :
        2d array time x output dims.
    eval_rates :
        2d array time x units
    eval_behavior :
        2d array time x output dims
    grid_search :
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        Uniform average R2 score on eval data
    """
    if np.any(np.isnan(train_behavior)):
        train_rates = train_rates[~np.isnan(train_behavior)[:, 0]]
        train_behavior = train_behavior[~np.isnan(train_behavior)[:, 0]]
    if np.any(np.isnan(eval_behavior)):
        eval_rates = eval_rates[~np.isnan(eval_behavior)[:, 0]]
        eval_behavior = eval_behavior[~np.isnan(eval_behavior)[:, 0]]
    assert not np.any(np.isnan(train_rates)) and not np.any(np.isnan(eval_rates)), \
        "fit_and_eval_decoder: NaNs found in rate predictions within required trial times"

    if grid_search:
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)})
    else:
        decoder = Ridge(alpha=1e-2)
    decoder.fit(train_rates, train_behavior)
    if return_preds:
        return decoder.score(eval_rates, eval_behavior), decoder, decoder.predict(eval_rates)
    else:
        return decoder.score(eval_rates, eval_behavior), decoder