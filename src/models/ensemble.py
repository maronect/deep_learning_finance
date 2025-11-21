def ensemble_predictions(mu_hist, mu_lr, mu_rnn, w_hist=0.6, w_lr=0.25, w_rnn=0.15):
    """
    Ensemble final do retorno esperado mensal.
    """
    return w_hist * mu_hist + w_lr * mu_lr + w_rnn * mu_rnn
