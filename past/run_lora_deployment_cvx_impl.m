function results = run_lora_deployment_cvx_impl()
    clc;
    close all;
    rng(12);

    if exist('cvx_begin', 'file') ~= 2
        error('CVX is not on the MATLAB path.');
    end

    params = default_parameters();
    state = build_scenario(params);
    results = solve_all_intervals(params, state);
    print_summary(results, params);
    plot_ao_convergence(results, params);
end


function params = default_parameters()
    params.N = 1;
    params.G = 2;
    params.U = 4;
    params.M = 6;

    params.tau = 1.00e5;
    params.tth = 10.0;
    params.B = 5.0e6;

    params.Kccs = 8;
    params.Kg = 8;
    params.Ku = 8;
    params.betaMax = 3;

    params.fCCS = 4.68e12;
    params.fUGV = 4.68e12;
    params.fUAV = 8.90e12;
    params.fup = 2.07e10;

    params.muCCS = 1.0;
    params.muUGV = 1.0;
    params.muUAV = 1.0;

    params.xiCCS = 179.0;
    params.xiUGV = 179.0;
    params.xiUAV = 336.1;

    params.pCCScomp = 260.0;
    params.pUGVcomp = 180.0;
    params.pUAVcomp = 110.0;
    params.PmaxCCS = 20.0;
    params.PmaxUGV = 8.0;

    params.dData = 15e6 * ones(params.N, 1);
    params.dCCS = 0.875 * 8e6 * ones(params.N, 1);
    params.dUGV = 15.0 * 8e6 * ones(params.N, 1);

    params.Fup = params.dData .* params.fup;
    params.FferUGV = 3.105e17 * ones(params.G, params.N);
    params.FferGU = 1.785e17 * ones(params.G, params.U, params.N);

    params.g0CCS = 1.0;
    params.g0UGV = 1.0;
    params.pathLossExpCCS = 2.2;
    params.pathLossExpUGV = 2.1;
    params.kappaCCS = 5.0;
    params.kappaUGV = 3.0;
    params.lambda = 0.125;
    params.antSpacing = params.lambda / 2.0;
    params.sigmaCCS2 = 2.0e-4;
    params.sigmaUAV2 = 1.5e-3;

    params.ccsPos = [0.0; 0.0; 80.0];
    params.uavHeight = 60.0;

    params.outerMaxIter = 12;
    params.outerMinIter = 8;
    params.outerTol = 1.0e-4;
    params.updateDamping = 0.35;

    params.s2MaxEval = 24;
    params.s2FeasTol = 1.0e-5;
    params.s6MaxIter = 20;
    params.s6Tol = 1.0e-4;

    params.epsVal = 1.0e-8;
    params.figName = 'ao_convergence_interval_1.png';
end


function state = build_scenario(params)
    state.ugvPos = zeros(2, params.G, params.N);
    state.uavPos = zeros(2, params.U, params.N);

    ugvBase = [-24.0, 24.0;
               -12.0, 14.0];
    ugvVel = [0.8, -0.6;
              0.4,  0.3];

    uavBase = [-33.0, -15.0, 15.0, 34.0;
                26.0,   8.0, 16.0, 32.0];
    uavVel = [0.6, -0.4, 0.5, -0.5;
              0.2,  0.3, 0.2,  0.1];

    for n = 1:params.N
        state.ugvPos(:, :, n) = ugvBase + (n - 1) * ugvVel;
        state.uavPos(:, :, n) = uavBase + (n - 1) * uavVel;
    end

    state.hCCS = cell(params.N, 1);
    state.hUGV = cell(params.N, 1);
    for n = 1:params.N
        state.hCCS{n} = generate_ccs_ugv_channels(params, state.ugvPos(:, :, n));
        state.hUGV{n} = generate_ugv_uav_channels(params, state.ugvPos(:, :, n), state.uavPos(:, :, n));
    end
end


function results = solve_all_intervals(params, state)
    totalEnergy = 0.0;
    results.intervals = cell(params.N, 1);
    for n = 1:params.N
        intervalState.hCCS = state.hCCS{n};
        intervalState.hUGV = state.hUGV{n};
        results.intervals{n} = solve_single_interval(params, intervalState, n);
        totalEnergy = totalEnergy + results.intervals{n}.totalEnergy;
    end
    results.averageEnergy = totalEnergy / params.N;
end


function sol = solve_single_interval(params, state, n)
    alphaCCS = params.Kccs;
    alphaG = params.Kg * ones(params.G, 1);
    alphaU = params.Ku * ones(params.U, 1);
    beta = initial_association(params);
    gamma = (1 / params.G) * ones(params.G, 1);
    p = 0.90 * params.PmaxUGV * ones(params.G, 1);
    W = build_initial_beamforming(params, state.hCCS);

    prevObj = inf;
    history = nan(params.outerMaxIter, 1);

    for iter = 1:params.outerMaxIter
        old.alphaCCS = alphaCCS;
        old.alphaG = alphaG;
        old.alphaU = alphaU;
        old.beta = beta;
        old.gamma = gamma;
        old.p = p;
        old.W = W;

        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        alphaCCS = solve_s1_alpha_ccs(params, n, metrics);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        Wcand = solve_s2_beamforming(params, state.hCCS, n, metrics);
        W = blend_matrix(old.W, Wcand, params.updateDamping);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        alphaG = solve_s3_alpha_ugv(params, n, metrics);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        beta = solve_s4_association(params, state.hUGV, n, metrics, gamma, p, alphaU);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        gammaCand = solve_s5_bandwidth(params, state.hUGV, n, metrics, beta, p, alphaU);
        gamma = blend_vector(old.gamma, gammaCand, params.updateDamping);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        pCand = solve_s6_power(params, state.hUGV, n, metrics, beta, gamma, alphaU, old.p);
        p = blend_vector(old.p, pCand, params.updateDamping);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        alphaU = solve_s7_alpha_uav(params, n, metrics, beta);
        metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

        obj = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, p, metrics);
        [accepted, alphaCCS, alphaG, alphaU, beta, gamma, p, W, metrics, obj] = ...
            enforce_monotonicity(params, state, n, old, alphaCCS, alphaG, alphaU, beta, gamma, p, W, metrics, obj, prevObj);

        history(iter) = obj;
        relGap = abs(prevObj - obj) / max(1.0, abs(prevObj));
        prevObj = obj;

        if iter >= params.outerMinIter && accepted && relGap <= params.outerTol
            break;
        end
    end

    history = history(~isnan(history));
    validate_interval_solution(params, state, n, alphaCCS, alphaG, alphaU, W, beta, gamma, p);
    metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);

    sol.alphaCCS = alphaCCS;
    sol.alphaUGV = alphaG;
    sol.alphaUAV = alphaU;
    sol.W = W;
    sol.beta = beta;
    sol.gamma = gamma;
    sol.p = p;
    sol.metrics = metrics;
    sol.totalEnergy = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, p, metrics);
    sol.history = history;
end


function [accepted, alphaCCS, alphaG, alphaU, beta, gamma, p, W, metrics, obj] = enforce_monotonicity( ...
    params, state, n, old, alphaCCS, alphaG, alphaU, beta, gamma, p, W, metrics, obj, prevObj)

    accepted = true;
    if isinf(prevObj) || obj <= prevObj + 1.0e-8
        return;
    end

    accepted = false;
    trialWeights = [0.25, 0.10, 0.05];
    for idx = 1:numel(trialWeights)
        lam = trialWeights(idx);
        Wtry = blend_matrix(old.W, W, lam);
        gammaTry = blend_vector(old.gamma, gamma, lam);
        pTry = blend_vector(old.p, p, lam);
        metricsTry = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gammaTry, pTry, Wtry);
        objTry = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, pTry, metricsTry);
        if objTry <= prevObj + 1.0e-8 && is_feasible_metrics(params, metricsTry, Wtry, beta, gammaTry, pTry, alphaCCS, alphaG, alphaU)
            accepted = true;
            W = Wtry;
            gamma = gammaTry;
            p = pTry;
            metrics = metricsTry;
            obj = objTry;
            return;
        end
    end

    alphaCCS = old.alphaCCS;
    alphaG = old.alphaG;
    alphaU = old.alphaU;
    beta = old.beta;
    gamma = old.gamma;
    p = old.p;
    W = old.W;
    metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);
    obj = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, p, metrics);
end


function metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W)
    metrics.lupdCCS = ccs_update_latency(params, n, alphaCCS);
    metrics.lupdG = ugv_update_latency(params, n, alphaG);
    metrics.uavUpd = zeros(params.U, 1);
    metrics.assocG = zeros(params.U, 1);
    metrics.txPowerCCS = real(trace(W));

    metrics.ccsRate = zeros(params.G, 1);
    metrics.ccsTx = zeros(params.G, 1);
    for g = 1:params.G
        snr = real(state.hCCS(:, g)' * W * state.hCCS(:, g)) / params.sigmaCCS2;
        metrics.ccsRate(g) = params.B * log2(1.0 + max(snr, 0.0));
        metrics.ccsTx(g) = params.dCCS(n) / max(metrics.ccsRate(g), params.epsVal);
    end

    metrics.userRate = zeros(params.U, 1);
    metrics.userTx = zeros(params.U, 1);
    for u = 1:params.U
        g = find(beta(:, u) > 0.5, 1, 'first');
        if isempty(g)
            error('Each UAV must be associated with exactly one UGV.');
        end
        metrics.assocG(u) = g;
        metrics.uavUpd(u) = uav_update_latency(params, n, g, u, alphaU(u));
        gainRatio = channel_gain_ratio(params, abs(state.hUGV(g, u))^2);
        metrics.userRate(u) = gamma(g) * params.B * log2(1.0 + gainRatio * p(g));
        metrics.userTx(u) = params.dUGV(n) / max(metrics.userRate(u), params.epsVal);
    end

    metrics.groupTxMax = zeros(params.G, 1);
    metrics.groupJointMax = zeros(params.G, 1);
    metrics.totalDelay = zeros(params.G, 1);
    for g = 1:params.G
        users = find(metrics.assocG == g);
        if isempty(users)
            metrics.groupTxMax(g) = 0.0;
            metrics.groupJointMax(g) = 0.0;
        else
            metrics.groupTxMax(g) = max(metrics.userTx(users));
            metrics.groupJointMax(g) = max(metrics.userTx(users) + metrics.uavUpd(users));
        end
        metrics.totalDelay(g) = metrics.lupdCCS + metrics.ccsTx(g) + metrics.lupdG(g) + metrics.groupJointMax(g);
    end
end


function alphaCCS = solve_s1_alpha_ccs(params, n, metrics)
    lowerBound = 1;
    for g = 1:params.G
        rhs = params.tau - metrics.ccsTx(g) - metrics.lupdG(g) - metrics.groupJointMax(g) - params.xiCCS;
        if rhs <= params.epsVal
            error('S1 infeasible: CCS delay budget is non-positive.');
        end
        lowerBound = max(lowerBound, ceil(params.muCCS * params.Fup(n) / (params.fCCS * rhs)));
    end
    alphaCCS = min(params.Kccs, lowerBound);
end


function Wbest = solve_s2_beamforming(params, hCCS, n, metrics)
    delayBudget = params.tau - metrics.lupdCCS - metrics.lupdG - metrics.groupJointMax;
    etaMax = min([params.tth; delayBudget]);
    if etaMax <= params.epsVal
        error('S2 infeasible: no admissible CCS transmission budget.');
    end

    etaLow = max(params.epsVal, 1.0e-2);
    etaHigh = etaMax;
    for iter = 1:30
        etaMid = 0.5 * (etaLow + etaHigh);
        if is_s2_feasible(params, hCCS, n, etaMid)
            etaHigh = etaMid;
        else
            etaLow = etaMid;
        end
    end

    objFun = @(eta) s2_objective_value(params, hCCS, n, eta);
    opts = optimset('Display', 'off', 'TolX', 1.0e-3, 'MaxFunEvals', params.s2MaxEval);
    etaStar = fminbnd(objFun, etaHigh, etaMax, opts);
    [~, Wbest] = s2_objective_value(params, hCCS, n, etaStar);
    if isempty(Wbest)
        [~, Wbest] = s2_objective_value(params, hCCS, n, etaHigh);
    end
    if isempty(Wbest)
        error('S2 failed: no feasible beamforming matrix found.');
    end
end


function feasible = is_s2_feasible(params, hCCS, n, eta)
    [value, ~] = s2_objective_value(params, hCCS, n, eta);
    feasible = isfinite(value);
end


function [value, Wopt] = s2_objective_value(params, hCCS, n, eta)
    Hm = cell(params.G, 1);
    for g = 1:params.G
        Hm{g} = hCCS(:, g) * hCCS(:, g)';
    end
    snrReq = params.sigmaCCS2 * (2.0^(params.dCCS(n) / (params.B * eta)) - 1.0);

    cvx_begin quiet sdp
        variable W(params.M, params.M) hermitian semidefinite
        minimize(real(trace(W)))
        subject to
            real(trace(W)) <= params.PmaxCCS;
            for g = 1:params.G
                real(trace(Hm{g} * W)) >= snrReq;
            end
    cvx_end

    if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
        value = inf;
        Wopt = [];
        return;
    end

    Wopt = W;
    rates = zeros(params.G, 1);
    for g = 1:params.G
        snr = real(hCCS(:, g)' * Wopt * hCCS(:, g)) / params.sigmaCCS2;
        rates(g) = params.B * log2(1.0 + max(snr, 0.0));
    end
    txLatency = params.dCCS(n) ./ max(rates, params.epsVal);
    if any(txLatency > eta + params.s2FeasTol)
        value = inf;
        Wopt = [];
        return;
    end
    value = real(trace(Wopt)) * max(txLatency);
end


function alphaG = solve_s3_alpha_ugv(params, n, metrics)
    alphaG = zeros(params.G, 1);
    for g = 1:params.G
        rhs = params.tau - metrics.lupdCCS - metrics.ccsTx(g) - metrics.groupJointMax(g) - params.xiUGV;
        if rhs <= params.epsVal
            error('S3 infeasible: UGV %d has no residual delay budget.', g);
        end
        alphaG(g) = min(params.Kg, max(1, ceil(params.muUGV * params.FferUGV(g, n) / (params.fUGV * rhs))));
    end
end


function beta = solve_s4_association(params, hUGV, n, metrics, gamma, p, alphaU)
    beta = [];
    bestObjective = inf;
    assignments = enumerate_assignments(params.G, params.U);
    baseDelay = metrics.lupdCCS + metrics.ccsTx + metrics.lupdG;

    txLatency = zeros(params.G, params.U);
    jointLatency = zeros(params.G, params.U);
    for g = 1:params.G
        for u = 1:params.U
            coeff = channel_gain_ratio(params, abs(hUGV(g, u))^2);
            rate = gamma(g) * params.B * log2(1.0 + coeff * p(g));
            txLatency(g, u) = params.dUGV(n) / max(rate, params.epsVal);
            jointLatency(g, u) = txLatency(g, u) + uav_update_latency(params, n, g, u, alphaU(u));
        end
    end

    for row = 1:size(assignments, 1)
        candidate = zeros(params.G, params.U);
        for u = 1:params.U
            candidate(assignments(row, u), u) = 1;
        end
        if any(sum(candidate, 2) > params.betaMax)
            continue;
        end

        feasible = true;
        objective = 0.0;
        for g = 1:params.G
            users = find(candidate(g, :) > 0.5);
            if isempty(users)
                txMax = 0.0;
                jointMax = 0.0;
            else
                txMax = max(txLatency(g, users));
                jointMax = max(jointLatency(g, users));
            end
            if baseDelay(g) + jointMax > params.tau + 1.0e-8
                feasible = false;
                break;
            end
            objective = objective + p(g) * txMax;
        end

        if feasible && objective < bestObjective - 1.0e-9
            bestObjective = objective;
            beta = candidate;
        end
    end

    if isempty(beta)
        error('S4 infeasible: no binary association satisfies constraints (20d)-(20f) and (20k).');
    end
end


function gamma = solve_s5_bandwidth(params, hUGV, n, metrics, beta, p, alphaU)
    baseDelay = metrics.lupdCCS + metrics.ccsTx + metrics.lupdG;

    cvx_begin quiet
        variable gammaVar(params.G)
        variable zTx(params.G)
        variable zJoint(params.G)
        minimize(sum(p .* zTx))
        subject to
            gammaVar >= 1.0e-3;
            gammaVar <= 1.0;
            sum(gammaVar) <= 1.0;
            zTx >= 0.0;
            zJoint >= 0.0;
            for g = 1:params.G
                users = find(beta(g, :) > 0.5);
                for idx = 1:numel(users)
                    u = users(idx);
                    gainRatio = channel_gain_ratio(params, abs(hUGV(g, u))^2);
                    kappa = params.dUGV(n) / max(params.B * log2(1.0 + gainRatio * p(g)), params.epsVal);
                    upd = uav_update_latency(params, n, g, u, alphaU(u));
                    zTx(g) >= kappa * inv_pos(gammaVar(g));
                    zJoint(g) >= kappa * inv_pos(gammaVar(g)) + upd;
                end
                baseDelay(g) + zJoint(g) <= params.tau;
            end
    cvx_end

    if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
        error('S5 failed: %s', cvx_status);
    end

    gamma = gammaVar;
end


function p = solve_s6_power(params, hUGV, n, metrics, beta, gamma, alphaU, ~)
    p = zeros(params.G, 1);
    baseDelay = metrics.lupdCCS + metrics.ccsTx + metrics.lupdG;

    for g = 1:params.G
        users = find(beta(g, :) > 0.5);
        if isempty(users)
            p(g) = 0.0;
            continue;
        end

        pLower = 0.0;
        for idx = 1:numel(users)
            u = users(idx);
            residual = params.tau - baseDelay(g) - uav_update_latency(params, n, g, u, alphaU(u));
            if residual <= params.epsVal
                error('S6 infeasible: UGV %d has no transmission budget for UAV %d.', g, u);
            end
            gainRatio = channel_gain_ratio(params, abs(hUGV(g, u))^2);
            spectralNeed = params.dUGV(n) / (gamma(g) * params.B * residual);
            pReq = (2.0^spectralNeed - 1.0) / max(gainRatio, params.epsVal);
            pLower = max(pLower, pReq);
        end

        if pLower > params.PmaxUGV + 1.0e-8
            error('S6 infeasible: required power exceeds Pmax for UGV %d.', g);
        end

        objective = @(pg) local_s6_objective(pg, params, hUGV, n, beta, gamma, g);
        if params.PmaxUGV - pLower <= 1.0e-6
            p(g) = pLower;
        else
            opts = optimset('Display', 'off', 'TolX', 1.0e-4);
            p(g) = fminbnd(objective, pLower, params.PmaxUGV, opts);
        end
    end
end


function val = local_s6_objective(pg, params, hUGV, n, beta, gamma, g)
    users = find(beta(g, :) > 0.5);
    txVals = zeros(numel(users), 1);
    for idx = 1:numel(users)
        u = users(idx);
        gainRatio = channel_gain_ratio(params, abs(hUGV(g, u))^2);
        rate = gamma(g) * params.B * log2(1.0 + gainRatio * pg);
        txVals(idx) = params.dUGV(n) / max(rate, params.epsVal);
    end
    val = pg * max(txVals);
end


function alphaU = solve_s7_alpha_uav(params, n, metrics, beta)
    alphaU = zeros(params.U, 1);
    for u = 1:params.U
        g = find(beta(:, u) > 0.5, 1, 'first');
        rhs = params.tau - metrics.lupdCCS - metrics.ccsTx(g) - metrics.lupdG(g) - metrics.userTx(u) - params.xiUAV;
        if rhs <= params.epsVal
            error('S7 infeasible: UAV %d has no residual delay budget.', u);
        end
        alphaU(u) = min(params.Ku, max(2, ceil(params.muUAV * params.FferGU(g, u, n) / (params.fUAV * rhs))));
    end
end


function val = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, p, metrics)
    eCCS = alphaCCS * params.pCCScomp * metrics.lupdCCS + metrics.txPowerCCS * max(metrics.ccsTx);
    eUGV = 0.0;
    for g = 1:params.G
        users = find(beta(g, :) > 0.5);
        if isempty(users)
            txMax = 0.0;
        else
            txMax = max(metrics.userTx(users));
        end
        eUGV = eUGV + alphaG(g) * params.pUGVcomp * metrics.lupdG(g) + p(g) * txMax;
    end
    eUAV = sum(alphaU .* params.pUAVcomp .* metrics.uavUpd);
    val = eCCS + eUGV + eUAV;
end


function val = ccs_update_latency(params, n, alphaCCS)
    val = params.muCCS * params.Fup(n) / (alphaCCS * params.fCCS) + params.xiCCS;
end


function val = ugv_update_latency(params, n, alphaG)
    val = zeros(params.G, 1);
    for g = 1:params.G
        val(g) = params.muUGV * params.FferUGV(g, n) / (alphaG(g) * params.fUGV) + params.xiUGV;
    end
end


function val = uav_update_latency(params, n, g, u, alphaU)
    val = params.muUAV * params.FferGU(g, u, n) / (alphaU * params.fUAV) + params.xiUAV;
end


function beta = initial_association(params)
    beta = zeros(params.G, params.U);
    for u = 1:params.U
        g = mod(u - 1, params.G) + 1;
        beta(g, u) = 1;
    end
end


function W = build_initial_beamforming(params, hCCS)
    W = zeros(params.M, params.M);
    for g = 1:params.G
        W = W + hCCS(:, g) * hCCS(:, g)';
    end
    W = params.PmaxCCS * W / max(real(trace(W)), params.epsVal);
end


function h = generate_ccs_ugv_channels(params, ugvPos)
    h = zeros(params.M, params.G);
    for g = 1:params.G
        pos = [ugvPos(:, g); 0.0];
        delta = pos - params.ccsPos;
        dist = norm(delta);
        pathloss = sqrt(params.g0CCS / (dist ^ params.pathLossExpCCS));
        angle = acos(params.ccsPos(3) / dist);
        phase = -1j * 2.0 * pi * params.antSpacing * cos(angle) * (0:(params.M - 1)).' / params.lambda;
        aLos = exp(phase) / sqrt(params.M);
        aNlos = (randn(params.M, 1) + 1j * randn(params.M, 1)) / sqrt(2.0 * params.M);
        h(:, g) = pathloss * (sqrt(params.kappaCCS / (params.kappaCCS + 1.0)) * aLos ...
            + sqrt(1.0 / (params.kappaCCS + 1.0)) * aNlos);
    end
end


function h = generate_ugv_uav_channels(params, ugvPos, uavPos)
    h = zeros(params.G, params.U);
    for g = 1:params.G
        ugv3 = [ugvPos(:, g); 0.0];
        for u = 1:params.U
            uav3 = [uavPos(:, u); params.uavHeight];
            dist = norm(uav3 - ugv3);
            pathloss = sqrt(params.g0UGV / (dist ^ params.pathLossExpUGV));
            nlos = (randn + 1j * randn) / sqrt(2.0);
            h(g, u) = pathloss * (sqrt(params.kappaUGV / (params.kappaUGV + 1.0)) ...
                + sqrt(1.0 / (params.kappaUGV + 1.0)) * nlos);
        end
    end
end


function assignments = enumerate_assignments(G, U)
    total = G ^ U;
    assignments = zeros(total, U);
    for idx = 0:(total - 1)
        value = idx;
        for u = 1:U
            assignments(idx + 1, u) = mod(value, G) + 1;
            value = floor(value / G);
        end
    end
end


function out = blend_vector(oldVal, newVal, weight)
    out = (1.0 - weight) * oldVal + weight * newVal;
end


function out = blend_matrix(oldVal, newVal, weight)
    out = (1.0 - weight) * oldVal + weight * newVal;
end


function ratio = channel_gain_ratio(params, absH2)
    ratio = absH2 / params.sigmaUAV2;
end


function ok = is_feasible_metrics(params, metrics, W, beta, gamma, p, alphaCCS, alphaG, alphaU)
    ok = true;
    ok = ok && alphaCCS >= 1 && alphaCCS <= params.Kccs;
    ok = ok && all(alphaG >= 1 & alphaG <= params.Kg);
    ok = ok && all(alphaU >= 2 & alphaU <= params.Ku);
    ok = ok && real(trace(W)) <= params.PmaxCCS + 1.0e-8;
    ok = ok && all(abs(sum(beta, 1) - 1.0) <= 1.0e-8);
    ok = ok && all(sum(beta, 2) <= params.betaMax + 1.0e-8);
    ok = ok && all(gamma > 0.0) && sum(gamma) <= 1.0 + 1.0e-8;
    ok = ok && all(p >= -1.0e-8 & p <= params.PmaxUGV + 1.0e-8);
    ok = ok && all(metrics.totalDelay <= params.tau + 1.0e-6);
    ok = ok && all(metrics.ccsTx <= params.tth + 1.0e-6);
end


function validate_interval_solution(params, state, n, alphaCCS, alphaG, alphaU, W, beta, gamma, p)
    metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);
    if ~is_feasible_metrics(params, metrics, W, beta, gamma, p, alphaCCS, alphaG, alphaU)
        error('Final solution violates the paper constraints.');
    end
end


function print_summary(results, params)
    interval = results.intervals{1};
    metrics = interval.metrics;

    fprintf('Average energy: %.4f J\n', results.averageEnergy);
    fprintf('alpha_CCS: %d\n', interval.alphaCCS);
    fprintf('alpha_UGV: [%s]\n', format_vec(interval.alphaUGV));
    fprintf('alpha_UAV: [%s]\n', format_vec(interval.alphaUAV));
    fprintf('beta:\n');
    disp(interval.beta);
    fprintf('gamma: [%s]\n', format_vec(interval.gamma));
    fprintf('p_UGV (W): [%s]\n', format_vec(interval.p));
    fprintf('CCS transmit latency (s): [%s]\n', format_vec(metrics.ccsTx));
    fprintf('UGV-UAV transmit latency (s): [%s]\n', format_vec(metrics.userTx));
    fprintf('Total group delay (s): [%s]\n', format_vec(metrics.totalDelay));
    fprintf('CCS power usage (W): %.4f\n', metrics.txPowerCCS);
    fprintf('AO history: [%s]\n', format_vec(interval.history));
end


function txt = format_vec(vec)
    txt = strtrim(sprintf('%.4f ', vec(:).'));
end


function plot_ao_convergence(results, params)
    interval = results.intervals{1};
    fig = figure('Visible', 'off');
    plot(1:numel(interval.history), interval.history, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    grid on;
    xlabel('AO iteration');
    ylabel('Total energy (J)');
    title('AO Convergence');
    legend('Interval 1', 'Location', 'northeast');
    saveas(fig, params.figName);
    close(fig);
end




