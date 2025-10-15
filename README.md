EDAT or entropy driven adaptive trading model

calculates 
ε (micro-noise): variance of 1-min |Δmid| or tick-imbalance volatility (upticks–downticks over 30–60s).
ζ (turbulence): realized vol or Parkinson (HL), plus HL-entropy (histogram entropy of HL ranges).
φ (info energy): volume × |Δprice|.
ω (cycle): short Welch PSD peak ratio or 10–40 bar Hilbert phase coherence.
η (adaptive weight): 1 / (1 + rolling σ_returns).
Aux: order-book pressure (Σ depth_ask – Σ depth_bid at top 5), CVD slope.

Latent state (ξ)
HMM with 3–4 states trained on [ε, ζ, φ, ω, book-pressure].

τ (state duration)
Run-length since last state change → feeds confidence and position cap. 

