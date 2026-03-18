% UCN Biological Validation: Phase-Locking Proof (MANUAL PHASE CHECK + VALUED SPIKES)
% Uses EXACT manual style:
% - RK4 phase update
% - if phase>=2*pi: r=max(0,tanh(input)); while phase>=2*pi: phase=phase-2*pi; spike_count++
% - explicit spike_event, spike_count_hist, valued output y(t)=spike_count*r
% - SPLV from bandpassed noisy LFP (downsample -> bandpass -> Hilbert)
% =============================================================
clear; clc; close all;

% 1) Setup Signal
T_duration = 1000;         % ms
dt = 0.1;                  % ms
time = 0:dt:T_duration;
steps = numel(time);

fs = 1000/dt;              % Hz  (dt in ms) -> dt=0.1 ms => fs=10 kHz

% Input: 8Hz theta rhythm + noise
freq_target  = 8;
clean_signal = sin(2*pi*freq_target*(time/1000));
noise_level  = 0.8;
input_LFP    = clean_signal + noise_level*randn(size(time));

% 2) UCN Parameters
Omega0   = (2*pi*freq_target)/1000;  % rad/ms
lambda   = .1;                       % leak
W_in     = 1.0;                      % input gain
theta_th = 2*pi;
alpha_th = 0.9;

% State + histories
theta = 0;                            % phase state (kept in [0,2pi) by manual subtraction)
history_theta = zeros(1, steps);

spike_event      = false(1, steps);   % event at time-step
spike_count_hist = zeros(1, steps);   % burst count at time-step

history_r   = zeros(1, steps);        % r(t)=max(0,tanh(u))
history_out = zeros(1, steps);        % y(t)=spike_count*r(t)

% 3) Simulation Loop (MANUAL PHASE CHECKING LIKE YOUR SNIPPET)
fprintf('Running UCN simulation (manual phase checking + valued spikes)...\n');

for t = 1:steps

    u_t = input_LFP(t) * W_in;   % input to neuron

    % --- RK4 Phase Dynamic Simulation (exact style) ---
    % phase_dyn(theta,input,Omega0,lambda) = -lambda*theta + Omega0 + input
    f1 = dt*phase_dyn(theta,            u_t, Omega0, lambda);
    f2 = dt*phase_dyn(theta + f1/2,     u_t, Omega0, lambda);
    f3 = dt*phase_dyn(theta + f2/2,     u_t, Omega0, lambda);
    f4 = dt*phase_dyn(theta + f3,       u_t, Omega0, lambda);
    theta = theta + (1/6)*(f1 + 2*f2 + 2*f3 + f4);

    % (optional but IMPORTANT to avoid "negative phase jumps" confusion)
    % Keep it manual & simple: phase never goes below 0
    if theta < 0
        theta = 0;
    end

    % --- Manual Firing Rule Enforcing (EXACT manner) ---
    spike_count = 0;

    if theta >= theta_th
        r_val = max(0, tanh(u_t));      % your valued spike magnitude rule

        while theta >= theta_th
            theta = theta - alpha_th*theta_th;   % manual wrapping by subtraction
            spike_count = spike_count + 1;
        end

    else
        r_val = 0;
    end

    % Store events/values
    spike_event(t)      = (spike_count > 0);
    spike_count_hist(t) = spike_count;

    history_r(t)   = r_val;
    history_out(t) = spike_count * r_val;

    history_theta(t) = theta;
end

% 4) Compute SPLV from bandpassed noisy LFP (stable)
bp_low  = 6;   % Hz
bp_high = 10;  % Hz

% Downsample to ~400 Hz to make 6-10 Hz filtering well-conditioned
q = round(fs/400);
q = max(q, 1);
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

% Spike times (events)
spike_idx = find(spike_event);
spike_t   = time(spike_idx);

% Map spike times to ds indices
spike_idx_ds_evt = round(spike_t / (dt*q)) + 1;
spike_idx_ds_evt = spike_idx_ds_evt(spike_idx_ds_evt>=1 & spike_idx_ds_evt<=numel(phase_ds));

% Expand phases by burst_count (same as before, but consistent with manual spike_count)
spike_phases = [];
for k = 1:numel(spike_idx)
    c = spike_count_hist(spike_idx(k));
    if c > 0
        id = round(spike_t(k) / (dt*q)) + 1;
        if id >= 1 && id <= numel(phase_ds)
            spike_phases = [spike_phases; repmat(phase_ds(id), c, 1)]; %#ok<AGROW>
        end
    end
end

if ~isempty(spike_phases)
    SPLV = abs(mean(exp(1i*spike_phases)));
    Nspk = numel(spike_phases);
else
    SPLV = 0;
    Nspk = 0;
end

[p_rayleigh, z_rayleigh] = rayleigh_test(spike_phases);
fprintf('SPLV (bandpassed noisy LFP): %.4f | N=%d | Rayleigh z=%.3f | p=%.3g\n', ...
    SPLV, Nspk, z_rayleigh, p_rayleigh);

% ========== B) JITTER SURROGATE (event-only; standard practice) ==========
% Randomly jitter each spike time by +/- jitter_ms, clamp to valid range.
% This preserves slow firing structure but destroys fine phase locking.

%===== PREP for Option B: define EVENT-ONLY spike times/phases =====
spike_idx_evt = find(spike_event);
spike_t_evt   = time(spike_idx_evt);

spike_idx_ds_evt_only = round(spike_t_evt/(dt*q)) + 1;
spike_idx_ds_evt_only = spike_idx_ds_evt_only(spike_idx_ds_evt_only>=1 & spike_idx_ds_evt_only<=numel(phase_ds));

spike_phases_evt = phase_ds(spike_idx_ds_evt_only);

if ~isempty(spike_phases_evt)
    SPLV_evt = abs(mean(exp(1i*spike_phases_evt)));
    N_evt = numel(spike_phases_evt);
else
    SPLV_evt = 0;
    N_evt = 0;
end

[p_ray_evt, z_ray_evt] = rayleigh_test(spike_phases_evt);
fprintf('EVENT-ONLY SPLV: %.4f | N=%d | Rayleigh z=%.3f | p=%.3g\n', ...
    SPLV_evt, N_evt, z_ray_evt, p_ray_evt);


if ~isempty(spike_t_evt) && ~isempty(spike_phases_evt)

    Nsurr = 2000;         % more = smoother p
    jitter_ms = 50;       % choose 30, 50, or 80
    SPLV_jit = zeros(Nsurr,1);

    L = numel(phase_ds);
    tmin = t_ds(1);
    tmax = t_ds(end);

    for s = 1:Nsurr
        jit = (2*rand(size(spike_t_evt))-1) * jitter_ms;   % uniform in [-jitter_ms, +jitter_ms]
        tJ  = spike_t_evt + jit;

        % clamp to valid time range
        tJ(tJ < tmin) = tmin;
        tJ(tJ > tmax) = tmax;

        idxJ = round(tJ/(dt*q)) + 1;
        idxJ(idxJ < 1) = 1;
        idxJ(idxJ > L) = L;

        phJ = phase_ds(idxJ);
        SPLV_jit(s) = abs(mean(exp(1i*phJ)));
    end

    p_emp_jit = mean(SPLV_jit >= SPLV_evt);

    fprintf('JITTER surrogate (±%d ms): mean=%.3f std=%.3f | emp p=%.4g (vs EVENT-ONLY SPLV=%.3f)\n', ...
        jitter_ms, mean(SPLV_jit), std(SPLV_jit), p_emp_jit, SPLV_evt);

    figure('Color','w','Position',[420 120 650 420]);
    histogram(SPLV_jit, 35); grid on;
    xline(SPLV_evt, 'LineWidth', 2);
    title(sprintf('Jitter surrogate (±%d ms) | observed=%.3f | emp p=%.3g', jitter_ms, SPLV_evt, p_emp_jit));
    xlabel('SPLV (event-only)'); ylabel('count');
end


% 5) Visualization (4 subplots)
figure('Color','w', 'Position', [100 60 1100 920]);

subplot(4,1,1);
plot(time, input_LFP, 'Color', [0.7 0.7 0.7]); hold on;
plot(time, clean_signal, 'k--', 'LineWidth', 1.8);
plot(t_ds, lfp_theta_ds, 'LineWidth', 1.2);
title('1) Input LFP: Noisy (gray) + Clean rhythm (black dashed) + Bandpassed theta (downsampled)');
xlim([0 1000]); grid on;
legend('Noisy LFP','Clean 8Hz','Bandpassed (6-10Hz)');

subplot(4,1,2);
plot(time, history_theta, 'b', 'LineWidth', 1.0); hold on;
yline(2*pi, 'g--', 'LineWidth', 1.2);
title('2) UCN Internal Phase \theta (manual subtract reset at 2\pi)');
ylabel('\theta'); xlim([0 1000]); grid on;
ylim([-0.2 6.6]);

subplot(4,1,3);
sig = lfp_theta_ds ./ (max(abs(lfp_theta_ds)) + eps);
plot(t_ds, sig, 'k', 'LineWidth', 1.2); hold on;
for k = 1:numel(spike_t)
    x_spk = spike_t(k);
    line([x_spk x_spk], [-1.5 1.5], 'Color', 'r', 'LineWidth', 0.6);
end
title(sprintf('3) Phase Locking Proof (EVENTS) | SPLV=%.3f | Rayleigh p=%.3g | N=%d', ...
    SPLV, p_rayleigh, Nspk));
ylabel('Theta-band (norm)'); xlim([0 1000]); ylim([-1.8 1.8]); grid on;

subplot(4,1,4);
plot(time, history_out, 'LineWidth', 1.1); grid on;
title('4) Valued Spike Output  y(t)=spike\_count(t)\times max(0,tanh(u(t)))');
xlabel('Time (ms)'); ylabel('y(t)');
xlim([0 1000]);

% 6) Polar histogram
figure('Color','w', 'Position', [380 120 720 620]);
if ~isempty(spike_phases)
    polarhistogram(spike_phases, 24, 'Normalization', 'probability'); hold on;
    v = mean(exp(1i*spike_phases));
    mu = angle(v);
    r  = abs(v);
    polarplot([mu mu], [0 r], 'LineWidth', 2);
    title(sprintf('Spike Phase Circular Histogram | SPLV=%.3f | p=%.3g', SPLV, p_rayleigh));
else
    text(0.1,0.5,'No spikes detected -> no phase histogram','FontSize',14);
    axis off;
end

% ===================== EXTRA ANALYSIS (ADDED ONLY) =====================
% Adds the extra stats/checks you had before:
% 7) preferred phase + bootstrap CI
% 8) matched surrogate test (circular shift) + histogram
% 9) spike-triggered average (STA)
% (No change to your simulation/tuning.)

% 7) Preferred phase + bootstrap CI (circular)
if ~isempty(spike_phases)
    v0 = mean(exp(1i*spike_phases));
    mu0 = angle(v0);
    mu_deg = mod(rad2deg(mu0), 360);

    B = 2000;
    mu_boot = zeros(B,1);
    N = numel(spike_phases);
    for b = 1:B
        idx = randi(N, [N,1]);
        vb = mean(exp(1i*spike_phases(idx)));
        mu_boot(b) = angle(vb);
    end

    % unwrap around mu0 then CI then wrap back
    mu_unw = unwrap([mu0; mu_boot]);
    mu_boot_unw = mu_unw(2:end);
    CI = prctile(mu_boot_unw, [2.5 97.5]);
    CI_deg = mod(rad2deg(CI), 360);

    fprintf('Preferred phase mu = %.1f deg | 95% CI [%.1f, %.1f] deg\n', ...
        mu_deg, CI_deg(1), CI_deg(2));
else
    fprintf('Preferred phase: no spikes -> not defined.\n');
end

% 8) Surrogate test: random circular shift of spike times (MATCHED to burst expansion)
% This keeps the LFP phase fixed and randomizes spike timing by circularly shifting
% the EVENT indices on the downsampled phase grid. We then expand phases by burst_count
% exactly like the observed SPLV computation (matched statistic).
if ~isempty(spike_t) && ~isempty(spike_phases)

    Nsurr = 1000;  % increase for more stable empirical p
    SPLV_surr = zeros(Nsurr,1);
    L = numel(phase_ds);

    % event spike indices on downsampled grid (one per EVENT)
    spike_idx_ds_evt2 = round(spike_t / (dt*q)) + 1;
    spike_idx_ds_evt2 = spike_idx_ds_evt2(spike_idx_ds_evt2>=1 & spike_idx_ds_evt2<=L);

    for s = 1:Nsurr
        shift = randi(L);

        ph_list = [];
        for k = 1:numel(spike_idx)  % spike_idx are event indices on original grid
            c = spike_count_hist(spike_idx(k));
            if c > 0
                id = round(spike_t(k) / (dt*q)) + 1;
                if id >= 1 && id <= L
                    id_s = id + shift;
                    while id_s > L
                        id_s = id_s - L;
                    end
                    ph_list = [ph_list; repmat(phase_ds(id_s), c, 1)]; %#ok<AGROW>
                end
            end
        end

        if ~isempty(ph_list)
            SPLV_surr(s) = abs(mean(exp(1i*ph_list)));
        else
            SPLV_surr(s) = 0;
        end
    end

    p_emp = mean(SPLV_surr >= SPLV);
    fprintf('Surrogate (matched) SPLV mean=%.3f std=%.3f | empirical p=%.4g\n', ...
        mean(SPLV_surr), std(SPLV_surr), p_emp);

    figure('Color','w','Position',[420 120 650 420]);
    histogram(SPLV_surr, 30); grid on;
    xline(SPLV, 'LineWidth', 2);
    title(sprintf('Matched surrogate SPLV distribution | observed=%.3f | emp p=%.3g', SPLV, p_emp));
    xlabel('SPLV'); ylabel('count');
end

% 9) Spike-triggered average (STA) on theta-band LFP (downsampled)
% Window +/- 150 ms around spike EVENT times (not expanded by burst count)
win_ms = 150;
win_samp = round((win_ms/1000)*fs_ds);
L = numel(lfp_theta_ds);

spk_evt_ds = round(spike_t/(dt*q)) + 1;
spk_evt_ds = spk_evt_ds(spk_evt_ds > win_samp & spk_evt_ds <= L-win_samp);

if ~isempty(spk_evt_ds)
    segs = zeros(numel(spk_evt_ds), 2*win_samp+1);
    for k = 1:numel(spk_evt_ds)
        i0 = spk_evt_ds(k);
        segs(k,:) = lfp_theta_ds(i0-win_samp:i0+win_samp);
    end
    sta = mean(segs,1);
    tau = (-win_samp:win_samp)/fs_ds * 1000; % ms

    figure('Color','w','Position',[450 160 700 420]);
    plot(tau, sta, 'LineWidth', 2); grid on;
    xline(0,'--','LineWidth',1.2);
    title('Spike-triggered average (STA) of theta-band LFP');
    xlabel('Time relative to spike (ms)'); ylabel('Amplitude (a.u.)');
else
    fprintf('STA: not enough spikes away from boundaries.\n');
end


% ========= Local Functions =========
function phase_dot = phase_dyn(phase, input, Omega0, lambda)
    phase_dot = -lambda .* phase + Omega0 + input;
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
