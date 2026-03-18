%% ======== REAL DATA: UCN vs LIF | REGIME SELECTOR (KEEP UCN CORE) ========
% Mouse17-130125 (Buzsaki-style dataset)
% + Added PPC (Pairwise Phase Consistency) for event phases (UCN/LIF)
%   PPC is unbiased w.r.t. spike count N (unlike PLV/SPLV).
% =========================================================================
clear; clc; close all;
rng(1);

% ---------------- USER SETTINGS ----------------
regimesToRun = {'WAKEepisode'};

% Epoch/snippet
min_epoch_s      = 30;      % only use episodes >= this length (seconds)
snippet_ms       = 1000;    % 1000 ms
snippet_margin_s = 0.5;     % snippet starts >= margin after epoch start
use_random_snip  = false;   % false: deterministic "start+margin". true: random inside epoch.

% Phase reference options
use_bandpass_for_phase = false; % thLFP is already theta-band
bp_low = 6; bp_high = 10;

% Surrogates
Nsurr_jitter  = 2000;
jitter_ms     = 50;
Nsurr_matched = 1000;       % circular shift surrogate

% STA
sta_win_ms = 150;

% ---------------- LOAD FILES ----------------
load('Mouse17-130125.SleepScoreLFP.LFP.mat','SleepScoreLFP');
load('Mouse17-130125.spikes.cellinfo.mat','spikes');           
load('Mouse17-130125.sessionInfo.mat','sessionInfo');
load('Mouse17-130125.SleepState.states.mat','SleepState');     
load('Mouse17-130125.SleepStateEpisodes.states.mat','SleepStateEpisodes');

% ---------------- SAFE UNWRAP ----------------
if isstruct(SleepScoreLFP) && isfield(SleepScoreLFP,'SleepScoreLFP'), SleepScoreLFP = SleepScoreLFP.SleepScoreLFP; end
if isstruct(sessionInfo)  && isfield(sessionInfo,'sessionInfo'),     sessionInfo  = sessionInfo.sessionInfo; end
if exist('SleepStateEpisodes','var') && isstruct(SleepStateEpisodes) && isfield(SleepStateEpisodes,'SleepStateEpisodes')
    SleepStateEpisodes = SleepStateEpisodes.SleepStateEpisodes;
end

% ---------------- BASIC SIGNAL ----------------
THid = SleepScoreLFP.THchanID;
SWid = SleepScoreLFP.SWchanID;

th_region = string(get_chan_region(sessionInfo, THid));
sw_region = string(get_chan_region(sessionInfo, SWid));

fprintf('\nTheta channel (THchanID=%d) region tag: %s\n', THid, th_region);
fprintf('Slow-wave channel (SWchanID=%d) region tag: %s\n', SWid, sw_region);

lfp_theta_full = double(SleepScoreLFP.thLFP(:));
t_sec          = double(SleepScoreLFP.t(:));
fs             = double(SleepScoreLFP.sf);
dt_ms          = 1000/fs;

if numel(t_sec) ~= numel(lfp_theta_full)
    t_sec = (0:numel(lfp_theta_full)-1)'/fs;
end

fprintf('\nLFP length = %d samples | fs=%.3f Hz | dt=%.4f ms | duration=%.1f s\n', ...
    numel(lfp_theta_full), fs, dt_ms, numel(lfp_theta_full)/fs);

% ---------------- OPTIONAL FILTER (for phase only) ----------------
if use_bandpass_for_phase
    d = designfilt('bandpassiir', ...
        'FilterOrder', 8, ...
        'HalfPowerFrequency1', bp_low, ...
        'HalfPowerFrequency2', bp_high, ...
        'SampleRate', fs, ...
        'DesignMethod', 'butter');
else
    d = [];
end

% ---------------- RUN REGIMES ----------------
for rr = 1:numel(regimesToRun)
    stateField = regimesToRun{rr};

    fprintf('\n====================================================\n');
    fprintf('RUNNING REGIME: %s\n', stateField);
    fprintf('====================================================\n');

    % ---- pick epoch interval ----
    [t0, t1, usedState] = pick_epoch_interval(SleepStateEpisodes, t_sec, stateField, min_epoch_s);
    fprintf('Using state="%s"\n', usedState);
    fprintf('Chosen epoch: [%.3f, %.3f] sec (len=%.2f s)\n', t0, t1, (t1-t0));

    % ---- pick snippet ----
    T_duration_s = snippet_ms/1000;
    if (t1 - t0) < T_duration_s
        warning('Epoch too short for snippet. Skipping %s.', stateField);
        continue;
    end

    if use_random_snip
        snippet_start = t0 + (t1 - t0 - T_duration_s)*rand();
    else
        snippet_start = t0 + snippet_margin_s;
        if snippet_start + T_duration_s > t1
            snippet_start = max(t0, t1 - T_duration_s - 0.1);
        end
    end
    snippet_end = snippet_start + T_duration_s;

    i0 = find(t_sec >= snippet_start, 1, 'first');
    i1 = find(t_sec <= snippet_end,   1, 'last');

    if isempty(i0) || isempty(i1) || i1 <= i0
        warning('Could not extract valid snippet. Skipping %s.', stateField);
        continue;
    end

    lfp_snip = lfp_theta_full(i0:i1);
    t_snip_sec = t_sec(i0:i1);
    steps = numel(lfp_snip);
    time_ms = (0:steps-1) * dt_ms;

    fprintf('Snippet: %.3f–%.3f sec | %d samples | %.1f ms\n', ...
        t_snip_sec(1), t_snip_sec(end), steps, time_ms(end));

    % ---- phase reference from theta-LFP snippet ----
    if use_bandpass_for_phase
        lfp_for_phase = filtfilt(d, lfp_snip);
    else
        lfp_for_phase = lfp_snip;
    end
    phase = angle(hilbert(lfp_for_phase));  % [-pi, pi]

    % =======================
    % UCN params (KEEP MANUAL LOGIC UNTOUCHED)
    % =======================
    freq_target = 8; % Hz nominal
    Omega0_UCN   = (2*pi*freq_target)/1500;  % rad/ms
    lambda_UCN   = 0.01;
    W_in_UCN     = 0.2;
    theta_th     = (2+0)*pi;
    alpha_th     = 0.7;

    theta = 0;
    ucn_theta = zeros(1,steps);
    ucn_spike_event = false(1,steps);
    ucn_spike_count = zeros(1,steps);
    ucn_r = zeros(1,steps);
    ucn_out = zeros(1,steps);

    input_LFP = zscore(lfp_snip(:))';  % normalize for driving models

    % ======= UCN CORE LOOP =======
    for t = 1:steps
        u_t = input_LFP(t) * W_in_UCN;

        f1 = dt_ms*phase_dyn(theta,            u_t, Omega0_UCN, lambda_UCN);
        f2 = dt_ms*phase_dyn(theta + f1/2,     u_t, Omega0_UCN, lambda_UCN);
        f3 = dt_ms*phase_dyn(theta + f2/2,     u_t, Omega0_UCN, lambda_UCN);
        f4 = dt_ms*phase_dyn(theta + f3,       u_t, Omega0_UCN, lambda_UCN);
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
        ucn_r(t)           = r_val;
        ucn_out(t)         = spike_count * r_val;
        ucn_theta(t)       = theta;
    end

    % =======================
    % LIF model  (KEEP V(t) FOR PLOT)
    % =======================
    Vrest   = -65; Vreset = -65; Vth = -50;
    tau_m   = 12; R = 10; tref = 2;
    I_bias  = 0.5; I_gain = 1.0;

    V = Vrest;
    ref_count = 0;
    lif_V = zeros(1,steps);
    lif_spike_event = false(1,steps);

    for t = 1:steps
        I_t = I_bias + I_gain*input_LFP(t);

        if ref_count > 0
            ref_count = ref_count - 1;
            V = Vreset;
            spk = 0;
        else
            dV = (-(V - Vrest) + R*I_t) / tau_m;
            V  = V + dt_ms*dV;

            if V >= Vth
                spk = 1;
                V = Vreset;
                ref_count = round(tref/dt_ms);
            else
                spk = 0;
            end
        end

        lif_V(t) = V;
        lif_spike_event(t) = (spk==1);
    end

    % =======================
    % Phase-locking stats (EVENT-ONLY) for UCN / LIF
    % =======================
    stats_UCN = compute_phase_stats_simple(ucn_spike_event, phase, time_ms);
    stats_LIF = compute_phase_stats_simple(lif_spike_event, phase, time_ms);

    % ======================= **HERE** =======================
    % Add PPC (pairwise phase consistency) to both stats structs
    stats_UCN.PPC = compute_ppc(stats_UCN.spike_phases);
    stats_LIF.PPC = compute_ppc(stats_LIF.spike_phases);
    % ========================================================

    fprintf('\n=== EVENT-ONLY PHASE LOCKING (%s) ===\n', usedState);
    print_stats('UCN', stats_UCN);
    fprintf('    PPC=%.4f\n', stats_UCN.PPC);
    print_stats('LIF', stats_LIF);
    fprintf('    PPC=%.4f\n', stats_LIF.PPC);

    % =======================
    % Jitter surrogate (EVENT-ONLY)
    % =======================
    stats_UCN = add_jitter_surrogate_simple(stats_UCN, phase, time_ms, Nsurr_jitter, jitter_ms);
    stats_LIF = add_jitter_surrogate_simple(stats_LIF, phase, time_ms, Nsurr_jitter, jitter_ms);

    fprintf('\n=== JITTER SURROGATE (±%d ms) empirical p ===\n', jitter_ms);
    fprintf('UCN: p=%.4g | mean=%.3f std=%.3f\n', stats_UCN.p_jit, mean(stats_UCN.SPLV_jit), std(stats_UCN.SPLV_jit));
    fprintf('LIF: p=%.4g | mean=%.3f std=%.3f\n', stats_LIF.p_jit, mean(stats_LIF.SPLV_jit), std(stats_LIF.SPLV_jit));

    figure('Color','w','Position',[420 120 650 420]);
    histogram(stats_UCN.SPLV_jit, 35); grid on; xline(stats_UCN.SPLV,'LineWidth',2);
    title(sprintf('%s | UCN jitter surrogate (±%d ms) | observed=%.3f | emp p=%.3g', usedState, jitter_ms, stats_UCN.SPLV, stats_UCN.p_jit));
    xlabel('SPLV (event-only)'); ylabel('count');

    % =======================
    % Matched surrogate: circular shift spike TIMES on the PHASE GRID (EVENT-ONLY)
    % =======================
    [SPLV_surrU, p_empU] = matched_shift_surrogate_eventonly(stats_UCN.spike_t, phase, time_ms, Nsurr_matched);
    fprintf('Matched surrogate (UCN EVENT-only): mean=%.3f std=%.3f | emp p=%.4g\n', mean(SPLV_surrU), std(SPLV_surrU), p_empU);

    figure('Color','w','Position',[420 120 650 420]);
    histogram(SPLV_surrU, 30); grid on; xline(stats_UCN.SPLV,'LineWidth',2);
    title(sprintf('%s | Matched surrogate SPLV (UCN) | observed=%.3f | emp p=%.3g', usedState, stats_UCN.SPLV, p_empU));
    xlabel('SPLV'); ylabel('count');

    [SPLV_surrL, p_empL] = matched_shift_surrogate_eventonly(stats_LIF.spike_t, phase, time_ms, Nsurr_matched);
    fprintf('Matched surrogate (LIF EVENT-only): mean=%.3f std=%.3f | emp p=%.4g\n', mean(SPLV_surrL), std(SPLV_surrL), p_empL);

    figure('Color','w','Position',[420 120 650 420]);
    histogram(SPLV_surrL, 30); grid on; xline(stats_LIF.SPLV,'LineWidth',2);
    title(sprintf('%s | Matched surrogate SPLV (LIF) | observed=%.3f | emp p=%.3g', usedState, stats_LIF.SPLV, p_empL));
    xlabel('SPLV'); ylabel('count');

    % =======================
    % ΔSPLV and ΔPPC
    % =======================
    dSPLV = stats_UCN.SPLV - stats_LIF.SPLV;
    dPPC  = stats_UCN.PPC  - stats_LIF.PPC;
    fprintf('\nΔSPLV (UCN - LIF) = %.4f\n', dSPLV);
    fprintf('ΔPPC  (UCN - LIF) = %.4f\n', dPPC);

    figure('Color','w','Position',[430 160 560 360]);
    bar([stats_UCN.SPLV, stats_LIF.SPLV; stats_UCN.PPC, stats_LIF.PPC]); grid on;
    ylim([0 1]);
    set(gca,'XTickLabel',{'SPLV','PPC'});
    legend({'UCN','LIF'},'Location','best');
    title(sprintf('%s | Event-only SPLV & PPC', usedState));
    ylabel('Value');

    % =======================
    % STA on theta-band LFP
    % =======================
    plot_sta(stats_UCN.spike_t, lfp_for_phase, fs, sta_win_ms, usedState);

    % =======================
    % Main 4-panel plot (KEEP LIF V(t))
    % =======================
    figure('Color','w','Position',[80 60 1250 900]);

    subplot(4,1,1);
    plot(time_ms, lfp_snip, 'k','LineWidth',1.0); grid on;
    title(sprintf('REAL \theta-LFP snippet (%s) | THchanID=%d (%s)', usedState, THid, th_region));
    ylabel('thLFP');

    subplot(4,1,2);
    plot(time_ms, ucn_theta, 'LineWidth',1.0); hold on; yline(2*pi,'--');
    grid on; ylim([-0.2 6.6]);
    title(sprintf('UCN \\theta(t) | SPLV=%.3f PPC=%.3f pRay=%.2g pJit=%.2g N=%d', ...
        stats_UCN.SPLV, stats_UCN.PPC, stats_UCN.p_ray, stats_UCN.p_jit, stats_UCN.N));

    subplot(4,1,3);
    plot(time_ms, lif_V, 'LineWidth',1.0); hold on; yline(Vth,'--');
    grid on;
    title(sprintf('LIF V(t) | SPLV=%.3f PPC=%.3f pRay=%.2g pJit=%.2g N=%d', ...
        stats_LIF.SPLV, stats_LIF.PPC, stats_LIF.p_ray, stats_LIF.p_jit, stats_LIF.N));
    ylabel('V (mV)');

    subplot(4,1,4);
    plot(time_ms, (lfp_snip./(max(abs(lfp_snip))+eps)), 'k','LineWidth',1.0); hold on;
    plot_event_lines(stats_UCN.spike_t, [-1.5 1.5], [1 0 0], 0.6);
    plot_event_lines(stats_LIF.spike_t, [-1.5 1.5], [0 0.4 1], 0.6);
    grid on; ylim([-1.8 1.8]);
    title('Events on \theta-LFP (UCN=red, LIF=blue)');
    xlabel('Time (ms)');

    % =======================
    % Valued output plot
    % =======================
    figure('Color','w','Position',[350 180 850 360]);
    bar(time_ms, ucn_out, 'LineWidth', 1.2); grid on;
    title(sprintf('%s | UCN valued output y(t)=spike\\_count(t)\\times max(0,tanh(u(t)))', usedState));
    xlabel('Time (ms)'); ylabel('y(t)');

    % =======================
    % Polar histograms (UCN/LIF)
    % =======================
    figure('Color','w','Position',[100 100 900 380]);

    subplot(1,2,1);
    polarhistogram(stats_UCN.spike_phases, 24, 'Normalization','probability'); hold on;
    polarplot([deg2rad(stats_UCN.mu_deg) deg2rad(stats_UCN.mu_deg)], [0 stats_UCN.SPLV], 'LineWidth',2);
    title(sprintf('UCN | SPLV=%.3f | PPC=%.3f | p=%.2g', stats_UCN.SPLV, stats_UCN.PPC, stats_UCN.p_ray));

    subplot(1,2,2);
    polarhistogram(stats_LIF.spike_phases, 24, 'Normalization','probability'); hold on;
    polarplot([deg2rad(stats_LIF.mu_deg) deg2rad(stats_LIF.mu_deg)], [0 stats_LIF.SPLV], 'LineWidth',2);
    title(sprintf('LIF | SPLV=%.3f | PPC=%.3f | p=%.2g', stats_LIF.SPLV, stats_LIF.PPC, stats_LIF.p_ray));
end

% =======================
% LOCAL FUNCTIONS
% =======================
function phase_dot = phase_dyn(phase, input, Omega0, lambda)
    phase_dot = -lambda .* phase + Omega0 + input;
end

function region = get_chan_region(sessionInfo, chanID)
    region = "";
    try
        if isfield(sessionInfo,'region') && numel(sessionInfo.region) >= chanID
            region = string(sessionInfo.region{chanID});
        end
    catch
        region = "";
    end
end

function [t0, t1, usedState] = pick_epoch_interval(SleepStateEpisodes, t_sec, desiredState, min_epoch_s)
    usedState = desiredState;
    if strcmpi(desiredState,'ALL')
        t0 = t_sec(1); t1 = t_sec(end);
        usedState = 'ALL';
        return;
    end

    ints = SleepStateEpisodes.ints;
    if ~isfield(ints, desiredState) || isempty(ints.(desiredState))
        warning('State "%s" not found or empty. Falling back to ALL.', desiredState);
        t0 = t_sec(1); t1 = t_sec(end);
        usedState = 'ALL';
        return;
    end

    epochs = ints.(desiredState);
    lens = epochs(:,2) - epochs(:,1);
    keep = find(lens >= min_epoch_s);
    if isempty(keep)
        warning('No epochs >= %.1fs for state "%s". Using first epoch anyway.', min_epoch_s, desiredState);
        pick_i = 1;
    else
        pick_i = keep(1);
    end
    t0 = epochs(pick_i,1);
    t1 = epochs(pick_i,2);
end

function stats = compute_phase_stats_simple(spike_event, phase, time_ms)
    spike_idx = find(spike_event);
    spike_t   = time_ms(spike_idx);
    spike_phases = phase(spike_idx);

    if ~isempty(spike_phases)
        SPLV = abs(mean(exp(1i*spike_phases)));
        N = numel(spike_phases);
    else
        SPLV = 0; N = 0;
    end

    [p_ray, z_ray] = rayleigh_test(spike_phases);

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

function stats = add_jitter_surrogate_simple(stats, phase, time_ms, Nsurr, jitter_ms)
    if isempty(stats.spike_t) || isempty(stats.spike_phases)
        stats.SPLV_jit = [];
        stats.p_jit = 1;
        return;
    end

    spike_t = stats.spike_t(:);
    dt = median(diff(time_ms));
    L = numel(phase);

    SPLV_jit = zeros(Nsurr,1);
    for s = 1:Nsurr
        jit = (2*rand(size(spike_t))-1) * jitter_ms;
        tJ  = spike_t + jit;

        idxJ = round(tJ/dt) + 1;
        idxJ(idxJ < 1) = 1;
        idxJ(idxJ > L) = L;

        phJ = phase(idxJ);
        SPLV_jit(s) = abs(mean(exp(1i*phJ)));
    end

    stats.SPLV_jit = SPLV_jit;
    stats.p_jit = mean(SPLV_jit >= stats.SPLV);
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

function PPC = compute_ppc(alpha)
    % Pairwise Phase Consistency (Vinck et al.)
    % Equivalent efficient formula:
    % PPC = (|sum(e^{i alpha})|^2 - N) / (N*(N-1))
    if isempty(alpha)
        PPC = NaN; return;
    end
    alpha = alpha(:);
    N = numel(alpha);
    if N < 2
        PPC = NaN; return;
    end
    S = sum(exp(1i*alpha));
    PPC = (abs(S)^2 - N) / (N*(N-1));
    % numerical safety
    PPC = max(min(real(PPC), 1), -1);
end

function print_stats(name, stats)
    fprintf('%s: SPLV=%.4f | N=%d | Rayleigh z=%.3f p=%.3g | mu=%.1f° CI[%.1f, %.1f]°\n', ...
        name, stats.SPLV, stats.N, stats.z_ray, stats.p_ray, stats.mu_deg, stats.CI_deg(1), stats.CI_deg(2));
end

function plot_event_lines(spike_t, ylims, rgb, lw)
    for k = 1:numel(spike_t)
        x = spike_t(k);
        line([x x], ylims, 'Color', rgb, 'LineWidth', lw);
    end
end

function [SPLV_surr, p_emp] = matched_shift_surrogate_eventonly(spike_t_ms, phase, time_ms, Nsurr)
    if isempty(spike_t_ms)
        SPLV_surr = zeros(Nsurr,1);
        p_emp = 1;
        return;
    end

    dt = median(diff(time_ms));
    L = numel(phase);

    idx_evt = round(spike_t_ms(:)/dt) + 1;
    idx_evt(idx_evt<1) = 1; idx_evt(idx_evt>L) = L;

    ph_obs = phase(idx_evt);
    SPLV_obs = abs(mean(exp(1i*ph_obs)));

    SPLV_surr = zeros(Nsurr,1);
    for s = 1:Nsurr
        shift = randi(L);
        idxS = idx_evt + shift;
        idxS(idxS > L) = idxS(idxS > L) - L;
        phS = phase(idxS);
        SPLV_surr(s) = abs(mean(exp(1i*phS)));
    end

    p_emp = mean(SPLV_surr >= SPLV_obs);
end

function plot_sta(spike_t_ms, lfp_theta, fs, win_ms, labelStr)
    if isempty(spike_t_ms), fprintf('STA: no spikes.\n'); return; end

    dt_ms = 1000/fs;
    spk_idx = round(spike_t_ms(:)/dt_ms) + 1;
    spk_idx(spk_idx<1) = 1;
    spk_idx(spk_idx>numel(lfp_theta)) = numel(lfp_theta);

    win_samp = round((win_ms/1000)*fs);
    L = numel(lfp_theta);

    spk_idx = spk_idx(spk_idx > win_samp & spk_idx <= L-win_samp);
    if isempty(spk_idx)
        fprintf('STA: spikes too close to boundaries.\n');
        return;
    end

    segs = zeros(numel(spk_idx), 2*win_samp+1);
    for k = 1:numel(spk_idx)
        i0 = spk_idx(k);
        segs(k,:) = lfp_theta(i0-win_samp:i0+win_samp);
    end
    sta = mean(segs,1);
    tau = (-win_samp:win_samp)/fs * 1000;

    figure('Color','w','Position',[450 160 700 420]);
    plot(tau, sta, 'LineWidth', 2); grid on;
    xline(0,'--','LineWidth',1.2);
    title(sprintf('%s | Spike-triggered average (STA) of theta-band LFP', labelStr));
    xlabel('Time relative to spike (ms)'); ylabel('Amplitude (a.u.)');
end
