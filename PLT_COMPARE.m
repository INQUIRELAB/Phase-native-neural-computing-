%% UCN vs LIF-SNN (same noisy LFP, same theta-phase analysis)  [ANN REMOVED]
% - Uses the SAME input signal and the SAME theta-band phase reference
% - Computes phase-locking metrics for UCN spikes and LIF spikes
% - Adds: EVENT-ONLY SPLV, Rayleigh test, jitter surrogate (±50 ms), preferred phase + bootstrap CI
% =============================================================
clear; clc; close all;

rng(1); % reproducibility

%% 1) Setup Signal (same as your UCN test)
T_duration = 1000;         % ms
dt = 0.1;                  % ms
time = 0:dt:T_duration;
steps = numel(time);

fs = 1000/dt;              % Hz  (dt in ms) -> dt=0.1 ms => fs=10 kHz

freq_target  = 8;          % Hz
clean_signal = sin(2*pi*freq_target*(time/1000));
noise_level  = 0.8;
input_LFP    = clean_signal + noise_level*randn(size(time));

%% 2) Theta-band phase reference from noisy LFP (stable)
bp_low  = 6;   % Hz
bp_high = 10;  % Hz

q = round(fs/400);  q = max(q,1);
fs_ds = fs/q;

lfp_ds = decimate(input_LFP, q);
t_ds   = time(1:q:end);

d = designfilt('bandpassiir', ...
    'FilterOrder', 8, ...
    'HalfPowerFrequency1', bp_low, ...
    'HalfPowerFrequency2', bp_high, ...
    'SampleRate', fs_ds, ...
    'DesignMethod', 'butter');

lfp_theta_ds = filtfilt(d, lfp_ds);
phase_ds = angle(hilbert(lfp_theta_ds)); % [-pi, pi]
Lds = numel(phase_ds); %#ok<NASGU>

%% =======================
% 3) MODEL #1: UCN (your manual phase-check + valued spikes)
%% =======================
Omega0_UCN   = (2*pi*freq_target)/1000;  % rad/ms
lambda_UCN   = 0.1;
W_in_UCN     = 1.0;
theta_th     = 2*pi;
alpha_th = 0.7;
theta = 0;
ucn_theta = zeros(1,steps);
ucn_spike_event = false(1,steps);
ucn_spike_count = zeros(1,steps);
ucn_r = zeros(1,steps);
ucn_out = zeros(1,steps);

for t = 1:steps
    u_t = input_LFP(t) * W_in_UCN;

    f1 = dt*phase_dyn(theta,            u_t, Omega0_UCN, lambda_UCN);
    f2 = dt*phase_dyn(theta + f1/2,     u_t, Omega0_UCN, lambda_UCN);
    f3 = dt*phase_dyn(theta + f2/2,     u_t, Omega0_UCN, lambda_UCN);
    f4 = dt*phase_dyn(theta + f3,       u_t, Omega0_UCN, lambda_UCN);
    theta = theta + (1/6)*(f1 + 2*f2 + 2*f3 + f4);

    if theta < 0
        theta = 0;
    end

    spike_count = 0;
    if theta >= theta_th
        r_val = max(0, tanh(u_t));
        while theta >= theta_th
            theta = theta - alpha_th*theta_th;
            spike_count = spike_count + 1;
        end
    else
        r_val = 0;
    end

    ucn_spike_event(t) = (spike_count > 0);
    ucn_spike_count(t) = spike_count;
    ucn_r(t) = r_val;
    ucn_out(t) = spike_count * r_val;
    ucn_theta(t) = theta;
end

%% =======================
% 4) MODEL #2: Standard Spiking Neuron (LIF)
%% =======================
% Classic current-driven LIF:
%   dV/dt = (-(V - Vrest) + R*I)/tau_m
% Spike when V>=Vth -> reset to Vreset, refractory

Vrest   = -65;    % mV
Vreset  = -65;    % mV
Vth     = -50;    % mV
tau_m   = 20;     % ms
R       = 10;     % MOhm (gain)
tref    = 2;      % ms refractory

% input current scaling (set so it spikes a reasonable amount)
I_bias  = 0.5;     % nA
I_gain  = 5.0;     % nA per input unit

V = Vrest;
ref_count = 0;

lif_V = zeros(1,steps);
lif_spike_event = false(1,steps);
lif_out = zeros(1,steps); % binary spikes (0/1)

for t = 1:steps
    I_t = I_bias + I_gain*input_LFP(t);

    if ref_count > 0
        ref_count = ref_count - 1;
        V = Vreset;
        spk = 0;
    else
        dV = (-(V - Vrest) + R*I_t) / tau_m; % mV/ms
        V  = V + dt*dV;

        if V >= Vth
            spk = 1;
            V = Vreset;
            ref_count = round(tref/dt);
        else
            spk = 0;
        end
    end

    lif_V(t) = V;
    lif_spike_event(t) = (spk==1);
    lif_out(t) = spk;
end

%% =======================
% 5) Compute phase-locking stats for each model (event-only)
%% =======================
stats_UCN = compute_phase_stats(ucn_spike_event, time, dt, q, phase_ds);
stats_LIF = compute_phase_stats(lif_spike_event, time, dt, q, phase_ds);

fprintf('\n=== EVENT-ONLY PHASE LOCKING (theta-band noisy LFP) ===\n');
print_stats('UCN', stats_UCN);
print_stats('LIF', stats_LIF);

%% =======================
% 6) Jitter surrogate (±50ms) for each model
%% =======================
Nsurr = 2000;
jitter_ms = 50;

stats_UCN = add_jitter_surrogate(stats_UCN, phase_ds, dt, q, Nsurr, jitter_ms);
stats_LIF = add_jitter_surrogate(stats_LIF, phase_ds, dt, q, Nsurr, jitter_ms);

fprintf('\n=== JITTER SURROGATE (±%d ms) empirical p ===\n', jitter_ms);
fprintf('UCN: p=%.4g | surrogate mean=%.3f std=%.3f\n', stats_UCN.p_jit, mean(stats_UCN.SPLV_jit), std(stats_UCN.SPLV_jit));
fprintf('LIF: p=%.4g | surrogate mean=%.3f std=%.3f\n', stats_LIF.p_jit, mean(stats_LIF.SPLV_jit), std(stats_LIF.SPLV_jit));

%% =======================
% 7) Visualization (shared input + per-model outputs)
%% =======================
figure('Color','w','Position',[80 60 1200 900]);

subplot(3,1,1);
plot(time, input_LFP, 'Color', [0.7 0.7 0.7]); hold on;
plot(time, clean_signal, 'k--', 'LineWidth', 1.6);
plot(t_ds, lfp_theta_ds, 'LineWidth', 1.2);
title('Input: Noisy LFP (gray) + Clean 8Hz (black dashed) + Bandpassed theta (downsampled)');
xlim([0 1000]); grid on; legend('Noisy','Clean','Bandpassed');

subplot(3,1,2);
plot(time, ucn_theta, 'LineWidth', 1.0); hold on;
yline(2*pi,'--','LineWidth',1.2);
title(sprintf('UCN: \\theta(t) + valued output | SPLV=%.3f pRay=%.1e pJit=%.3g N=%d', ...
    stats_UCN.SPLV, stats_UCN.p_ray, stats_UCN.p_jit, stats_UCN.N));
ylabel('\theta'); xlim([0 1000]); grid on; ylim([-0.2 6.6]);

subplot(3,1,3);
plot(time, lif_V, 'LineWidth', 1.0); hold on;
yline(Vth,'--','LineWidth',1.2);
title(sprintf('LIF: V(t) + spikes | SPLV=%.3f pRay=%.1e pJit=%.3g N=%d', ...
    stats_LIF.SPLV, stats_LIF.p_ray, stats_LIF.p_jit, stats_LIF.N));
ylabel('V (mV)'); xlabel('Time (ms)'); xlim([0 1000]); grid on;

%% Overlay event markers on theta-band for each model (separate figure)
figure('Color','w','Position',[120 80 1200 650]);

sig = lfp_theta_ds ./ (max(abs(lfp_theta_ds))+eps);

subplot(2,1,1);
plot(t_ds, sig, 'k','LineWidth',1.2); hold on;
plot_event_lines(stats_UCN.spike_t, [-1.5 1.5]);
title(sprintf('UCN events on theta-band | mu=%.1f° CI[%.1f, %.1f]°', stats_UCN.mu_deg, stats_UCN.CI_deg(1), stats_UCN.CI_deg(2)));
xlim([0 1000]); ylim([-1.8 1.8]); grid on;

subplot(2,1,2);
plot(t_ds, sig, 'k','LineWidth',1.2); hold on;
plot_event_lines(stats_LIF.spike_t, [-1.5 1.5]);
title(sprintf('LIF events on theta-band | mu=%.1f° CI[%.1f, %.1f]°', stats_LIF.mu_deg, stats_LIF.CI_deg(1), stats_LIF.CI_deg(2)));
xlim([0 1000]); ylim([-1.8 1.8]); grid on;

%% Polar histograms (UCN vs LIF)
figure('Color','w','Position',[150 120 900 420]);

subplot(1,2,1);
polarhistogram(stats_UCN.spike_phases, 24, 'Normalization','probability'); hold on;
polarplot([deg2rad(stats_UCN.mu_deg) deg2rad(stats_UCN.mu_deg)], [0 stats_UCN.SPLV], 'LineWidth',2);
title(sprintf('UCN | SPLV=%.3f', stats_UCN.SPLV));

subplot(1,2,2);
polarhistogram(stats_LIF.spike_phases, 24, 'Normalization','probability'); hold on;
polarplot([deg2rad(stats_LIF.mu_deg) deg2rad(stats_LIF.mu_deg)], [0 stats_LIF.SPLV], 'LineWidth',2);
title(sprintf('LIF | SPLV=%.3f', stats_LIF.SPLV));

%% Jitter surrogate distributions (optional but useful)
figure('Color','w','Position',[180 140 900 420]);

subplot(1,2,1);
histogram(stats_UCN.SPLV_jit, 35); grid on; xline(stats_UCN.SPLV,'LineWidth',2);
title(sprintf('UCN jitter (±%dms) | p=%.3g', jitter_ms, stats_UCN.p_jit)); xlabel('SPLV'); ylabel('count');

subplot(1,2,2);
histogram(stats_LIF.SPLV_jit, 35); grid on; xline(stats_LIF.SPLV,'LineWidth',2);
title(sprintf('LIF jitter (±%dms) | p=%.3g', jitter_ms, stats_LIF.p_jit)); xlabel('SPLV'); ylabel('count');

%% =======================
% Local Functions
%% =======================
function phase_dot = phase_dyn(phase, input, Omega0, lambda)
    phase_dot = -lambda .* phase + Omega0 + input;
end

function stats = compute_phase_stats(spike_event, time, dt, q, phase_ds)
    % event times
    spike_idx = find(spike_event);
    spike_t   = time(spike_idx);

    % map to downsampled indices
    idx_ds = round(spike_t/(dt*q)) + 1;
    idx_ds = idx_ds(idx_ds>=1 & idx_ds<=numel(phase_ds));

    spike_phases = phase_ds(idx_ds);

    if ~isempty(spike_phases)
        SPLV = abs(mean(exp(1i*spike_phases)));
        N = numel(spike_phases);
    else
        SPLV = 0; N = 0;
    end

    [p_ray, z_ray] = rayleigh_test(spike_phases);

    % preferred phase + bootstrap CI
    if ~isempty(spike_phases)
        v = mean(exp(1i*spike_phases));
        mu = angle(v);
        mu_deg = mod(rad2deg(mu), 360);

        B = 2000;
        mu_boot = zeros(B,1);
        for b = 1:B
            ii = randi(N,[N 1]);
            vb = mean(exp(1i*spike_phases(ii)));
            mu_boot(b) = angle(vb);
        end
        mu_unw = unwrap([mu; mu_boot]);
        CI = prctile(mu_unw(2:end), [2.5 97.5]);
        CI_deg = mod(rad2deg(CI), 360);
    else
        mu_deg = NaN; CI_deg = [NaN NaN];
    end

    stats.SPLV = SPLV;
    stats.N = N;
    stats.p_ray = p_ray;
    stats.z_ray = z_ray;
    stats.spike_t = spike_t;
    stats.spike_phases = spike_phases;
    stats.mu_deg = mu_deg;
    stats.CI_deg = CI_deg;
end

function stats = add_jitter_surrogate(stats, phase_ds, dt, q, Nsurr, jitter_ms)
    if isempty(stats.spike_t) || isempty(stats.spike_phases)
        stats.SPLV_jit = [];
        stats.p_jit = 1;
        return;
    end

    spike_t = stats.spike_t;
    L = numel(phase_ds);

    SPLV_jit = zeros(Nsurr,1);

    for s = 1:Nsurr
        jit = (2*rand(size(spike_t))-1) * jitter_ms;  % uniform jitter
        tJ  = spike_t + jit;

        idxJ = round(tJ/(dt*q)) + 1;
        idxJ(idxJ < 1) = 1;
        idxJ(idxJ > L) = L;

        phJ = phase_ds(idxJ);
        SPLV_jit(s) = abs(mean(exp(1i*phJ)));
    end

    p_emp = mean(SPLV_jit >= stats.SPLV);

    stats.SPLV_jit = SPLV_jit;
    stats.p_jit = p_emp;
end

function [p, z] = rayleigh_test(alpha)
    if isempty(alpha)
        p = 1; z = 0; return;
    end
    alpha = alpha(:);
    N = numel(alpha);
    R = abs(sum(exp(1i*alpha)));
    Rbar = R / N;
    z = N * (Rbar^2);

    p = exp(-z) * (1 + (2*z - z^2)/(4*N) - ...
        (24*z - 132*z^2 + 76*z^3 - 9*z^4)/(288*N^2));
    p = max(min(p,1),0);
end

function print_stats(name, stats)
    fprintf('%s: SPLV=%.4f | N=%d | Rayleigh z=%.3f p=%.3g | mu=%.1f° CI[%.1f, %.1f]°\n', ...
        name, stats.SPLV, stats.N, stats.z_ray, stats.p_ray, stats.mu_deg, stats.CI_deg(1), stats.CI_deg(2));
end

function plot_event_lines(spike_t, ylims)
    for k = 1:numel(spike_t)
        x = spike_t(k);
        line([x x], ylims, 'Color','r', 'LineWidth',0.6);
    end
end
