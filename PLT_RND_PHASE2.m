% UCN vs LIF: Nonstationary Phase-Diffusion Stress Test (SPLV curve + Lock fraction + Recovery)
% Produces:
%  (A) SPLV vs sigma_phi (mean±std)
%  (B) Locking-time fraction vs sigma_phi (mean±std)
%  (C) Recovery time after slips vs sigma_phi (mean±std)
%
% Uses SAME theta-band reference: downsample -> bandpass -> Hilbert
% Uses UCN manual spike logic: if theta>=2*pi -> r=max(0,tanh(u)); while theta>=2*pi: theta-=2*pi; spike_count++
% Uses LIF spikes for event times.
% =============================================================
clear; clc; close all;
rng(1);

% ======== USER: keep your UCN tuning EXACTLY here ========
T_duration = 1000;   % ms
dt = 0.1;            % ms
time = 0:dt:T_duration;
steps = numel(time);
fs = 1000/dt;        % Hz

freq_target = 8;     % Hz (nominal theta carrier)

% UCN params (DO NOT CHANGE)
Omega0_UCN = (2*pi*freq_target)/1000; % rad/ms
lambda_UCN = 0.1;
W_in_UCN   = 1.0;
theta_th   = 2*pi;

% LIF params (can stay)
Vrest = -65; Vreset = -65; Vth = -50;
tau_m = 20; R = 10; tref = 2; % ms
I_bias = 0.5; I_gain = 5.0;

% ======== Nonstationary phase diffusion sweep ========
% sigma_phi in rad/sqrt(s) (common way to parameterize phase random walk)
sigma_list = [0 5 10 15 20 25 30];  % adjust as you like
nTrials = 30;                       % Monte Carlo per sigma

% Windowed SPLV(t) settings (nonstationary-friendly)
win_ms = 250;     % window length
hop_ms = 25;      % hop
SPLV_lock_thr = 0.6;    % "locked" threshold (reasonable)
min_lock_ms = 100;      % require at least this long lock to count
min_lock_bins = ceil(min_lock_ms / hop_ms);

% Phase reference extraction settings
bp_low = 6; bp_high = 10;
q = round(fs/400); q = max(q,1);
fs_ds = fs/q;

% Preallocate results
res = struct();
for m = ["UCN","LIF"]
    res.(m).SPLV_mean = zeros(size(sigma_list));
    res.(m).SPLV_std  = zeros(size(sigma_list));
    res.(m).LockFrac_mean = zeros(size(sigma_list));
    res.(m).LockFrac_std  = zeros(size(sigma_list));
    res.(m).Rec_ms_mean = zeros(size(sigma_list));
    res.(m).Rec_ms_std  = zeros(size(sigma_list));
end

fprintf('Running sweep: %d sigmas x %d trials...\n', numel(sigma_list), nTrials);

% ======== Main sweep ========
for si = 1:numel(sigma_list)
    sigma_phi = sigma_list(si);

    SPLV_all_U = zeros(nTrials,1);
    SPLV_all_L = zeros(nTrials,1);
    Lock_all_U = zeros(nTrials,1);
    Lock_all_L = zeros(nTrials,1);
    Rec_all_U  = nan(nTrials,1);
    Rec_all_L  = nan(nTrials,1);

    for tr = 1:nTrials
        % ----- 1) Generate nonstationary input: phase random-walk carrier + additive noise -----
        % phi evolves: dphi/dt = w0 + sigma * eta(t)
        % where eta is white noise; sigma_phi given in rad/sqrt(s)
        % Convert to per-step noise on phi:
        dt_s = dt/1000;              % seconds
        w0 = 2*pi*freq_target;       % rad/s
        phi = zeros(1,steps);
        for t = 2:steps
            phi(t) = phi(t-1) + w0*dt_s + sigma_phi*sqrt(dt_s)*randn();
        end
        underlying = sin(phi);

        noise_level = 0.8; % same as your tests
        input_LFP = underlying + noise_level*randn(1,steps);

        % ----- 2) Phase reference: downsample -> bandpass -> Hilbert -----
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

        % ----- 3) UCN simulation (manual) -----
        theta = 0;
        ucn_event = false(1,steps);

        for t = 1:steps
            u_t = input_LFP(t) * W_in_UCN;

            f1 = dt*phase_dyn(theta,        u_t, Omega0_UCN, lambda_UCN);
            f2 = dt*phase_dyn(theta+f1/2,   u_t, Omega0_UCN, lambda_UCN);
            f3 = dt*phase_dyn(theta+f2/2,   u_t, Omega0_UCN, lambda_UCN);
            f4 = dt*phase_dyn(theta+f3,     u_t, Omega0_UCN, lambda_UCN);
            theta = theta + (1/6)*(f1 + 2*f2 + 2*f3 + f4);

            if theta < 0
                theta = 0;
            end

            spike_count = 0;
            if theta >= theta_th
                r_val = max(0, tanh(u_t));
                while theta >= theta_th
                    theta = theta - theta_th;
                    spike_count = spike_count + 1;
                end
            end

            ucn_event(t) = (spike_count > 0); % EVENT-ONLY for stats
        end

        % ----- 4) LIF simulation -----
        V = Vrest; ref_count = 0;
        lif_event = false(1,steps);

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
            lif_event(t) = (spk==1);
        end

        % ----- 5) Compute EVENT-ONLY SPLV + windowed SPLV(t) + lock fraction + recovery -----
        stU = compute_phase_stats_eventonly(ucn_event, time, dt, q, phase_ds);
        stL = compute_phase_stats_eventonly(lif_event, time, dt, q, phase_ds);

        % Windowed SPLV(t)
        [t_mid, splvU_t] = windowed_splv(ucn_event, time, dt, q, phase_ds, win_ms, hop_ms);
        [~,     splvL_t] = windowed_splv(lif_event, time, dt, q, phase_ds, win_ms, hop_ms);

        % Lock fraction
        Lock_all_U(tr) = mean(splvU_t >= SPLV_lock_thr);
        Lock_all_L(tr) = mean(splvL_t >= SPLV_lock_thr);

        % Recovery after slip:
        % define "locked" segments as >=thr for >=min_lock_bins consecutive windows
        Rec_all_U(tr) = recovery_time_ms(splvU_t, t_mid, SPLV_lock_thr, min_lock_bins);
        Rec_all_L(tr) = recovery_time_ms(splvL_t, t_mid, SPLV_lock_thr, min_lock_bins);

        SPLV_all_U(tr) = stU.SPLV;
        SPLV_all_L(tr) = stL.SPLV;
    end

    % Aggregate
    res.UCN.SPLV_mean(si) = mean(SPLV_all_U);
    res.UCN.SPLV_std(si)  = std(SPLV_all_U);

    res.LIF.SPLV_mean(si) = mean(SPLV_all_L);
    res.LIF.SPLV_std(si)  = std(SPLV_all_L);

    res.UCN.LockFrac_mean(si) = mean(Lock_all_U);
    res.UCN.LockFrac_std(si)  = std(Lock_all_U);

    res.LIF.LockFrac_mean(si) = mean(Lock_all_L);
    res.LIF.LockFrac_std(si)  = std(Lock_all_L);

    res.UCN.Rec_ms_mean(si) = mean(Rec_all_U,'omitnan');
    res.UCN.Rec_ms_std(si)  = std(Rec_all_U,'omitnan');

    res.LIF.Rec_ms_mean(si) = mean(Rec_all_L,'omitnan');
    res.LIF.Rec_ms_std(si)  = std(Rec_all_L,'omitnan');

    fprintf('sigma=%g: UCN SPLV %.3f±%.3f | LIF SPLV %.3f±%.3f\n', sigma_phi, ...
        res.UCN.SPLV_mean(si), res.UCN.SPLV_std(si), res.LIF.SPLV_mean(si), res.LIF.SPLV_std(si));
end

% ======== Plot A: SPLV vs sigma_phi ========
figure('Color','w','Position',[120 80 900 420]);
errorbar(sigma_list, res.UCN.SPLV_mean, res.UCN.SPLV_std, 'LineWidth', 2); hold on;
errorbar(sigma_list, res.LIF.SPLV_mean, res.LIF.SPLV_std, 'LineWidth', 2);
grid on;
xlabel('\sigma_\phi (rad/\surd s)'); ylabel('EVENT-ONLY SPLV');
title('SPLV vs phase-diffusion strength (nonstationary input)');
legend('UCN','LIF','Location','best');

% ======== Plot B: Locking time fraction vs sigma_phi ========
figure('Color','w','Position',[140 110 900 420]);
errorbar(sigma_list, res.UCN.LockFrac_mean, res.UCN.LockFrac_std, 'LineWidth', 2); hold on;
errorbar(sigma_list, res.LIF.LockFrac_mean, res.LIF.LockFrac_std, 'LineWidth', 2);
yline(SPLV_lock_thr,':'); 
grid on;
xlabel('\sigma_\phi (rad/\surd s)'); ylabel('Locking time fraction');
title(sprintf('Fraction of time windows with SPLV(t) \\ge %.2f (win=%dms, hop=%dms)', ...
    SPLV_lock_thr, win_ms, hop_ms));
legend('UCN','LIF','Location','best');

% ======== Plot C: Recovery time vs sigma_phi ========
figure('Color','w','Position',[160 140 900 420]);
errorbar(sigma_list, res.UCN.Rec_ms_mean, res.UCN.Rec_ms_std, 'LineWidth', 2); hold on;
errorbar(sigma_list, res.LIF.Rec_ms_mean, res.LIF.Rec_ms_std, 'LineWidth', 2);
grid on;
xlabel('\sigma_\phi (rad/\surd s)'); ylabel('Recovery time after slips (ms)');
title(sprintf('Recovery time to re-lock (\\ge %.2f for \\ge %d hops)', SPLV_lock_thr, min_lock_bins));
legend('UCN','LIF','Location','best');

% ======== Optional: show one example run (pick a sigma) ========
% Comment out if you want it lean
sigma_show = sigma_list(min(3,numel(sigma_list))); % e.g. 10
example_nonstationary_demo(sigma_show, time, dt, fs, freq_target, bp_low, bp_high, ...
    Omega0_UCN, lambda_UCN, W_in_UCN, theta_th, ...
    Vrest, Vreset, Vth, tau_m, R, tref, I_bias, I_gain, q, fs_ds);

% ===================== Local Functions =====================
function phase_dot = phase_dyn(phase, input, Omega0, lambda)
    phase_dot = -lambda .* phase + Omega0 + input;
end

function st = compute_phase_stats_eventonly(spike_event, time, dt, q, phase_ds)
    spike_idx = find(spike_event);
    spike_t   = time(spike_idx);

    idx_ds = round(spike_t/(dt*q)) + 1;
    idx_ds = idx_ds(idx_ds>=1 & idx_ds<=numel(phase_ds));
    spike_phases = phase_ds(idx_ds);

    if ~isempty(spike_phases)
        st.SPLV = abs(mean(exp(1i*spike_phases)));
        st.N = numel(spike_phases);
    else
        st.SPLV = 0; st.N = 0;
    end
end

function [t_mid, splv_t] = windowed_splv(spike_event, time, dt, q, phase_ds, win_ms, hop_ms)
    % window over original time axis; map spike times into ds for phases
    t0 = time(1); t1 = time(end);
    win = win_ms;
    hop = hop_ms;

    t_starts = t0:hop:(t1-win);
    nW = numel(t_starts);
    splv_t = zeros(nW,1);
    t_mid  = zeros(nW,1);

    for w = 1:nW
        a = t_starts(w);
        b = a + win;
        t_mid(w) = (a+b)/2;

        idx = find(spike_event & (time>=a) & (time<b));
        if isempty(idx)
            splv_t(w) = 0;
            continue;
        end
        spike_t = time(idx);

        idx_ds = round(spike_t/(dt*q)) + 1;
        idx_ds = idx_ds(idx_ds>=1 & idx_ds<=numel(phase_ds));
        if isempty(idx_ds)
            splv_t(w) = 0;
            continue;
        end
        ph = phase_ds(idx_ds);
        splv_t(w) = abs(mean(exp(1i*ph)));
    end
end

function rec_ms = recovery_time_ms(splv_t, t_mid, thr, min_bins)
    % Recovery time: after each slip (locked->unlocked), measure time until next stable lock
    % "stable lock" means >=thr for >=min_bins consecutive windows.
    if isempty(splv_t)
        rec_ms = NaN; return;
    end

    locked = splv_t >= thr;

    % Find stable locked segments
    stable = false(size(locked));
    run = 0;
    for i = 1:numel(locked)
        if locked(i), run = run+1; else, run = 0; end
        if run >= min_bins
            stable(i-min_bins+1:i) = true;
        end
    end

    % A "slip" is when stable goes from true to false
    slips = find(stable(1:end-1)==1 & stable(2:end)==0) + 1;
    if isempty(slips)
        rec_ms = NaN; return;
    end

    recs = nan(numel(slips),1);
    for s = 1:numel(slips)
        i0 = slips(s);
        % find next index where stable becomes true again
        j = find(stable(i0:end), 1, 'first');
        if isempty(j), continue; end
        i1 = i0 + j - 1;
        recs(s) = t_mid(i1) - t_mid(i0);
    end
    rec_ms = mean(recs,'omitnan');
end

function example_nonstationary_demo(sigma_phi, time, dt, fs, freq_target, bp_low, bp_high, ...
    Omega0_UCN, lambda_UCN, W_in_UCN, theta_th, ...
    Vrest, Vreset, Vth, tau_m, R, tref, I_bias, I_gain, q, fs_ds)

    steps = numel(time);
    dt_s = dt/1000;
    w0 = 2*pi*freq_target;

    phi = zeros(1,steps);
    for t = 2:steps
        phi(t) = phi(t-1) + w0*dt_s + sigma_phi*sqrt(dt_s)*randn();
    end
    underlying = sin(phi);

    noise_level = 0.8;
    input_LFP = underlying + noise_level*randn(1,steps);

    % phase reference
    lfp_ds = decimate(input_LFP, q);
    t_ds   = time(1:q:end);

    d = designfilt('bandpassiir', ...
        'FilterOrder', 8, ...
        'HalfPowerFrequency1', bp_low, ...
        'HalfPowerFrequency2', bp_high, ...
        'SampleRate', fs_ds, ...
        'DesignMethod', 'butter');

    lfp_theta_ds = filtfilt(d, lfp_ds);
    phase_ds = angle(hilbert(lfp_theta_ds));

    % UCN
    theta = 0;
    ucn_event = false(1,steps);
    ucn_theta = zeros(1,steps);

    for t = 1:steps
        u_t = input_LFP(t) * W_in_UCN;

        f1 = dt*phase_dyn(theta,        u_t, Omega0_UCN, lambda_UCN);
        f2 = dt*phase_dyn(theta+f1/2,   u_t, Omega0_UCN, lambda_UCN);
        f3 = dt*phase_dyn(theta+f2/2,   u_t, Omega0_UCN, lambda_UCN);
        f4 = dt*phase_dyn(theta+f3,     u_t, Omega0_UCN, lambda_UCN);
        theta = theta + (1/6)*(f1 + 2*f2 + 2*f3 + f4);

        if theta < 0, theta = 0; end

        spike_count = 0;
        if theta >= theta_th
            r_val = max(0, tanh(u_t)); 
            while theta >= theta_th
                theta = theta - theta_th;
                spike_count = spike_count + 1;
            end
        end
        ucn_event(t) = (spike_count>0);
        ucn_theta(t) = theta;
    end

    % LIF
    V = Vrest; ref_count = 0;
    lif_event = false(1,steps);
    lif_V = zeros(1,steps);

    for t = 1:steps
        I_t = I_bias + I_gain*input_LFP(t);

        if ref_count > 0
            ref_count = ref_count - 1;
            V = Vreset;
            spk = 0;
        else
            dV = (-(V - Vrest) + R*I_t)/tau_m;
            V = V + dt*dV;
            if V >= Vth
                spk = 1;
                V = Vreset;
                ref_count = round(tref/dt);
            else
                spk = 0;
            end
        end
        lif_event(t) = (spk==1);
        lif_V(t) = V;
    end

    stU = compute_phase_stats_eventonly(ucn_event, time, dt, q, phase_ds);
    stL = compute_phase_stats_eventonly(lif_event, time, dt, q, phase_ds);

    figure('Color','w','Position',[80 80 1100 800]);
    subplot(4,1,1);
    plot(time, input_LFP, 'Color',[.7 .7 .7]); hold on;
    plot(time, underlying, 'k--','LineWidth',1.2);
    plot(t_ds, lfp_theta_ds, 'LineWidth',1.2);
    title(sprintf('Example nonstationary input (phase RW)  \\sigma_\\phi=%.1f rad/\\surd s', sigma_phi));
    xlim([0 1000]); grid on; legend('Noisy LFP','Underlying','Bandpassed');

    subplot(4,1,2);
    plot(time, ucn_theta, 'LineWidth',1.0); hold on; yline(2*pi,'--');
    title(sprintf('UCN \\theta(t) | SPLV=%.3f (N=%d)', stU.SPLV, stU.N));
    xlim([0 1000]); ylim([-0.2 6.6]); grid on;

    subplot(4,1,3);
    plot(time, lif_V, 'LineWidth',1.0); hold on; yline(Vth,'--');
    title(sprintf('LIF V(t) | SPLV=%.3f (N=%d)', stL.SPLV, stL.N));
    xlim([0 1000]); grid on;

    subplot(4,1,4);
    [t_mid, splvU_t] = windowed_splv(ucn_event, time, dt, q, phase_ds, 250, 25);
    [~,     splvL_t] = windowed_splv(lif_event, time, dt, q, phase_ds, 250, 25);
    plot(t_mid, splvU_t,'LineWidth',1.8); hold on;
    plot(t_mid, splvL_t,'LineWidth',1.8);
    yline(0.6,'--');
    title('Windowed SPLV(t) (250ms window, 25ms hop)');
    xlabel('Time (ms)'); ylabel('SPLV(t)'); grid on; xlim([0 1000]);
    legend('UCN','LIF','Lock thr');
end
