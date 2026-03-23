function results = run_lora_deployment_cvx_v1()
    clc;
    close all;
    clear
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

    params.tau = 12.0;
    params.tth = 1.8;
    params.B = 24e6;

    params.Kccs = 12;
    params.Kg = 8;
    params.Ku = 6;

    params.fCCS = 55e9;
    params.fUGV = 18e9;
    params.fUAV = 10e9;

    params.muCCS = 1.0;
    params.muUGV = 1.0;
    params.muUAV = 1.0;

    params.xiCCS = 0.18;
    params.xiUGV = 0.10;
    params.xiUAV = 0.08;

    params.pCCScomp = 1.8;
    params.pUGVcomp = 1.0;
    params.pUAVcomp = 0.5;

    params.PmaxCCS = 8.0;
    params.PmaxUGV = 3.6;
    params.betaMax = 3;

    params.dData = 4.5e8 * ones(params.N, 1);
    params.dCCS = 1.6e7 * ones(params.N, 1);
    params.dUGV = 4.8e6 * ones(params.N, 1);

    params.Fup = 3.2e10 * ones(params.N, 1);
    params.FferCCS = 2.2e10 * ones(params.N, 1);
    params.FferUGV = repmat([8.0e9; 7.0e9], 1, params.N);

    params.FferGU = zeros(params.G, params.U, params.N);
    params.FferGU(:, :, 1) = [2.4e9, 2.2e9, 2.0e9, 1.8e9;
                              2.1e9, 2.3e9, 1.9e9, 2.2e9];
    params.g0 = 1.0;
    params.pathLossExpCCS = 2.2;
    params.pathLossExpUGV = 2.3;
    params.kappaCCS = 5.0;
    params.kappaUGV = 3.0;
    params.lambda = 0.125;
    params.antSpacing = params.lambda / 2.0;
    params.sigmaCCS2 = 1e-11;
    params.sigmaUAV2 = 5e-12;

    params.ccsPos = [0.0; 0.0; 80.0];
    params.uavHeight = 65.0;

    params.outerMaxIter = 16;
    params.outerMinIter = 8;
    params.outerTol = 5e-4;
    params.updateDamping = 0.72;

    params.s2MaxEval = 18;
    params.s2FeasTol = 1e-4;
    params.s6MaxIter = 20;
    params.s6Tol = 1e-4;

    params.epsVal = 1e-7;
    params.figName = 'ao_convergence_interval_1_v1.png';
end


function state = build_scenario(params)
    state.ugvPos = zeros(2, params.G, params.N);
    state.uavPos = zeros(2, params.U, params.N);

    ugvBase = [-28.0, 30.0;
               -18.0, 14.0];
    ugvVel = [1.4, -0.8;
              0.6,  0.5];

    uavBase = [-38.0, -12.0, 24.0, 50.0;
                28.0,  40.0, 18.0, 30.0];
    uavVel = [0.9, -0.5, 0.4, -0.7;
              0.3,  0.5, -0.2, 0.3];

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
    alphaCCS = 4;
    alphaG = 3 * ones(params.G, 1);
    alphaU = 2 * ones(params.U, 1);
    beta = initial_association(params);
    gamma = [0.54; 0.46];
    p = 0.80 * params.PmaxUGV * ones(params.G, 1);
    W = build_initial_beamforming(params, state.hCCS, n);

    prevObj = inf;
    history = zeros(params.outerMaxIter, 1);

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

        Wcand = solve_s2_beamforming(params, state.hCCS, n, metrics, W);
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

        if iter >= params.outerMinIter && relGap <= params.outerTol && accepted
            history = history(1:iter);
            break;
        end
    end

    history = history(history > 0);
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
    if isinf(prevObj) || obj <= prevObj + 1e-8
        return;
    end

    accepted = false;
    lambdas = [0.45, 0.25, 0.10];
    for k = 1:numel(lambdas)
        lam = lambdas(k);
        Wtry = blend_matrix(old.W, W, lam);
        gammaTry = blend_vector(old.gamma, gamma, lam);
        pTry = blend_vector(old.p, p, lam);
        mTry = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gammaTry, pTry, Wtry);
        objTry = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, pTry, mTry);
        if objTry <= prevObj + 1e-8 && is_feasible_metrics(params, mTry, Wtry, beta, gammaTry, pTry, alphaCCS, alphaG, alphaU)
            accepted = true;
            W = Wtry;
            gamma = gammaTry;
            p = pTry;
            metrics = mTry;
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
        metrics.ccsRate(g) = params.B * log2(1 + max(snr, 0));
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
        coeff = gamma(g) * params.B * log2(1 + channel_gain_ratio(params, abs(state.hUGV(g, u))^2) * p(g));
        metrics.userRate(u) = max(coeff, params.epsVal);
        metrics.userTx(u) = params.dUGV(n) / metrics.userRate(u);
    end

    metrics.groupTxMax = zeros(params.G, 1);
    metrics.groupJointMax = zeros(params.G, 1);
    metrics.totalDelay = zeros(params.G, 1);
    for g = 1:params.G
        users = find(metrics.assocG == g);
        if isempty(users)
            metrics.groupTxMax(g) = 0;
            metrics.groupJointMax(g) = 0;
        else
            metrics.groupTxMax(g) = max(metrics.userTx(users));
            metrics.groupJointMax(g) = max(metrics.userTx(users) + metrics.uavUpd(users));
        end
        metrics.totalDelay(g) = metrics.lupdCCS + metrics.ccsTx(g) + metrics.lupdG(g) + metrics.groupJointMax(g);
    end
end


function alphaCCS = solve_s1_alpha_ccs(params, n, metrics)
    A = (params.Fup(n) + params.FferCCS(n)) * params.muCCS / params.fCCS;
    lowerBound = 1;
    for g = 1:params.G
        rhs = params.tau - metrics.ccsTx(g) - metrics.lupdG(g) - metrics.groupJointMax(g) - params.xiCCS;
        if rhs <= params.epsVal
            error('S1 infeasible: CCS delay budget is non-positive.');
        end
        lowerBound = max(lowerBound, ceil(A / rhs));
    end
    alphaCCS = min(params.Kccs, lowerBound);
end


function Wbest = solve_s2_beamforming(params, hCCS, n, metrics, Wref)
    delayBudget = params.tau - metrics.lupdCCS - metrics.lupdG - metrics.groupJointMax;
    etaUpper = min([params.tth; delayBudget]);
    if etaUpper <= params.epsVal
        error('S2 infeasible: no positive CCS transmission budget.');
    end

    etaLower = locate_s2_lower_bound(params, hCCS, n, etaUpper, Wref);
    objective = @(eta) s2_objective_value(params, hCCS, n, eta, Wref);
    opts = optimset('Display', 'off', 'TolX', 5e-3, 'MaxFunEvals', params.s2MaxEval);
    etaStar = fminbnd(objective, etaLower, etaUpper, opts);
    [~, Wbest] = s2_objective_value(params, hCCS, n, etaStar, Wref);
    if isempty(Wbest)
        [~, Wbest] = s2_objective_value(params, hCCS, n, etaUpper, Wref);
    end
    if isempty(Wbest)
        error('S2 failed: no feasible beamforming solution found.');
    end
end


function etaLower = locate_s2_lower_bound(params, hCCS, n, etaUpper, Wref)
    lo = max(params.epsVal, 0.02);
    hi = etaUpper;
    for iter = 1:25
        mid = 0.5 * (lo + hi);
        value = s2_objective_value(params, hCCS, n, mid, Wref);
        if isfinite(value)
            hi = mid;
        else
            lo = mid;
        end
    end
    etaLower = hi;
end


function [value, Wopt] = s2_objective_value(params, hCCS, n, eta, Wref)
    G = params.G;
    M = params.M;
    Hm = cell(G, 1);
    for g = 1:G
        Hm{g} = hCCS(:, g) * hCCS(:, g)';
    end

    snrReq = params.sigmaCCS2 * (2^(params.dCCS(n) / (params.B * eta)) - 1);
    cvx_begin quiet sdp
        variable W(M, M) hermitian semidefinite
        minimize(real(trace(W)))
        subject to
            real(trace(W)) <= params.PmaxCCS;
            for g = 1:G
                real(trace(Hm{g} * W)) >= snrReq;
            end
    cvx_end

    if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
        value = inf;
        Wopt = [];
        return;
    end

    Wopt = W;
    txPower = real(trace(Wopt));
    rates = zeros(G, 1);
    for g = 1:G
        snr = real(hCCS(:, g)' * Wopt * hCCS(:, g)) / params.sigmaCCS2;
        rates(g) = params.B * log2(1 + max(snr, 0));
    end
    if any(params.dCCS(n) ./ max(rates, params.epsVal) > eta + params.s2FeasTol)
        value = inf;
        Wopt = [];
        return;
    end
    value = txPower * eta;
end


function alphaG = solve_s3_alpha_ugv(params, n, metrics)
    alphaG = zeros(params.G, 1);
    for g = 1:params.G
        A = params.FferUGV(g, n) * params.muUGV / params.fUGV;
        rhs = params.tau - metrics.lupdCCS - metrics.ccsTx(g) - metrics.groupJointMax(g) - params.xiUGV;
        if rhs <= params.epsVal
            error('S3 infeasible: UGV %d has no update budget.', g);
        end
        alphaG(g) = min(params.Kg, max(1, ceil(A / rhs)));
    end
end


function beta = solve_s4_association(params, hUGV, n, metrics, gamma, p, alphaU)
    beta = [];
    bestObjective = inf;
    allAssignments = enumerate_assignments(params.G, params.U);

    baseDelay = metrics.lupdCCS + metrics.ccsTx + metrics.lupdG;
    txLatency = zeros(params.G, params.U);
    jointLatency = zeros(params.G, params.U);
    for g = 1:params.G
        for u = 1:params.U
            rate = gamma(g) * params.B * log2(1 + channel_gain_ratio(params, abs(hUGV(g, u))^2) * p(g));
            txLatency(g, u) = params.dUGV(n) / max(rate, params.epsVal);
            jointLatency(g, u) = txLatency(g, u) + uav_update_latency(params, n, g, u, alphaU(u));
        end
    end

    for idx = 1:size(allAssignments, 1)
        assign = allAssignments(idx, :).';
        candidate = zeros(params.G, params.U);
        for u = 1:params.U
            candidate(assign(u), u) = 1;
        end

        if any(sum(candidate, 2) > params.betaMax)
            continue;
        end

        feasible = true;
        obj = 0.0;
        for g = 1:params.G
            users = find(candidate(g, :) > 0.5);
            if isempty(users)
                txMax = 0;
                jointMax = 0;
            else
                txMax = max(txLatency(g, users));
                jointMax = max(jointLatency(g, users));
            end
            if baseDelay(g) + jointMax > params.tau + 1e-8
                feasible = false;
                break;
            end
            obj = obj + p(g) * txMax;
        end

        if feasible && obj < bestObjective - 1e-9
            bestObjective = obj;
            beta = candidate;
        end
    end

    if isempty(beta)
        error('S4 infeasible: no binary association satisfies the paper constraints.');
    end
end


function gamma = solve_s5_bandwidth(params, hUGV, n, metrics, beta, p, alphaU)
    G = params.G;
    baseDelay = metrics.lupdCCS + metrics.ccsTx + metrics.lupdG;

    cvx_begin quiet
        variable gammaVar(G)
        variable zTx(G)
        variable zJoint(G)
        minimize(sum(p .* zTx))
        subject to
            gammaVar >= 1e-3;
            gammaVar <= 1;
            sum(gammaVar) <= 1;
            zTx >= 0;
            zJoint >= 0;
            for g = 1:G
                users = find(beta(g, :) > 0.5);
                for kk = 1:numel(users)
                    u = users(kk);
                    coeff = params.B * log2(1 + channel_gain_ratio(params, abs(hUGV(g, u))^2) * p(g));
                    kappa = params.dUGV(n) / max(coeff, params.epsVal);
                    lUpd = uav_update_latency(params, n, g, u, alphaU(u));
                    zTx(g) >= kappa * inv_pos(gammaVar(g));
                    zJoint(g) >= kappa * inv_pos(gammaVar(g)) + lUpd;
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
            p(g) = 0;
            continue;
        end

        pLower = 0;
        for kk = 1:numel(users)
            u = users(kk);
            upd = uav_update_latency(params, n, g, u, alphaU(u));
            remain = params.tau - baseDelay(g) - upd;
            if remain <= params.epsVal
                error('S6 infeasible: UAV %d leaves no transmission time at UGV %d.', u, g);
            end
            coeff = channel_gain_ratio(params, abs(hUGV(g, u))^2);
            reqRate = params.dUGV(n) / remain;
            pReq = (2^(reqRate / (gamma(g) * params.B)) - 1) / max(coeff, params.epsVal);
            pLower = max(pLower, pReq);
        end

        if pLower > params.PmaxUGV + 1e-8
            error('S6 infeasible: required power exceeds Pmax for UGV %d.', g);
        end

        objective = @(pg) local_s6_objective(pg, params, hUGV, n, beta, gamma, g);
        if params.PmaxUGV - pLower <= 1e-6
            p(g) = pLower;
        else
            opts = optimset('Display', 'off', 'TolX', 1e-4);
            p(g) = fminbnd(objective, pLower, params.PmaxUGV, opts);
        end
    end
end


function val = local_s6_objective(pg, params, hUGV, n, beta, gamma, g)
    users = find(beta(g, :) > 0.5);
    txVals = zeros(numel(users), 1);
    for kk = 1:numel(users)
        u = users(kk);
        coeff = channel_gain_ratio(params, abs(hUGV(g, u))^2);
        rate = gamma(g) * params.B * log2(1 + coeff * pg);
        txVals(kk) = params.dUGV(n) / max(rate, params.epsVal);
    end
    val = pg * max(txVals);
end

function alphaU = solve_s7_alpha_uav(params, n, metrics, beta)
    alphaU = zeros(params.U, 1);
    for u = 1:params.U
        g = find(beta(:, u) > 0.5, 1, 'first');
        A = params.FferGU(g, u, n) * params.muUAV / params.fUAV;
        rhs = params.tau - metrics.lupdCCS - metrics.ccsTx(g) - metrics.lupdG(g) - metrics.userTx(u) - params.xiUAV;
        if rhs <= params.epsVal
            error('S7 infeasible: UAV %d has no update budget.', u);
        end
        alphaU(u) = min(params.Ku, max(1, ceil(A / rhs)));
    end
end


function val = total_interval_energy(params, alphaCCS, alphaG, alphaU, beta, p, metrics)
    eCCS = alphaCCS * params.pCCScomp * metrics.lupdCCS + metrics.txPowerCCS * max(metrics.ccsTx);
    eUGV = 0;
    for g = 1:params.G
        users = find(beta(g, :) > 0.5);
        if isempty(users)
            txMax = 0;
        else
            txMax = max(metrics.userTx(users));
        end
        eUGV = eUGV + alphaG(g) * params.pUGVcomp * metrics.lupdG(g) + p(g) * txMax;
    end
    eUAV = sum(alphaU .* params.pUAVcomp .* metrics.uavUpd);
    val = eCCS + eUGV + eUAV;
end


function val = ccs_update_latency(params, n, alphaCCS)
    A = (params.Fup(n) + params.FferCCS(n)) * params.muCCS / params.fCCS;
    val = A / alphaCCS + params.xiCCS;
end


function val = ugv_update_latency(params, n, alphaG)
    val = zeros(params.G, 1);
    for g = 1:params.G
        A = params.FferUGV(g, n) * params.muUGV / params.fUGV;
        val(g) = A / alphaG(g) + params.xiUGV;
    end
end


function val = uav_update_latency(params, n, g, u, alphaU)
    A = params.FferGU(g, u, n) * params.muUAV / params.fUAV;
    val = A / alphaU + params.xiUAV;
end


function beta = initial_association(params)
    beta = zeros(params.G, params.U);
    for u = 1:params.U
        g = mod(u - 1, params.G) + 1;
        beta(g, u) = 1;
    end
end


function W = build_initial_beamforming(params, hCCS, n)
    etaCandidates = linspace(params.tth, max(0.20, 0.45 * params.tth), 16);
    W = [];
    bestValue = inf;

    for k = 1:numel(etaCandidates)
        eta0 = etaCandidates(k);
        [value, Wcand] = s2_objective_value(params, hCCS, n, eta0, zeros(params.M));
        if ~isempty(Wcand) && isfinite(value) && value < bestValue
            bestValue = value;
            W = Wcand;
        end
    end

    if isempty(W)
        error('Failed to initialize the CCS beamforming matrix. Increase tth or PmaxCCS, or relax the CCS channel/noise settings.');
    end
end

function h = generate_ccs_ugv_channels(params, ugvPos)
    h = zeros(params.M, params.G);
    idx = (0:(params.M - 1)).';
    for g = 1:params.G
        pos = [ugvPos(:, g); 0.0];
        diffVec = pos - params.ccsPos;
        dist = norm(diffVec);
        pathLoss = sqrt(params.g0 / dist^params.pathLossExpCCS);
        cosAod = (params.ccsPos(3) - pos(3)) / dist;
        steering = exp(-1i * 2 * pi * params.antSpacing / params.lambda * idx * cosAod) / sqrt(params.M);
        nlos = (randn(params.M, 1) + 1i * randn(params.M, 1)) / sqrt(2 * params.M);
        h(:, g) = pathLoss * (sqrt(params.kappaCCS / (params.kappaCCS + 1)) * steering + ...
            sqrt(1 / (params.kappaCCS + 1)) * nlos);
    end
end


function h = generate_ugv_uav_channels(params, ugvPos, uavPos)
    h = zeros(params.G, params.U);
    for g = 1:params.G
        ugv3 = [ugvPos(:, g); 0.0];
        for u = 1:params.U
            uav3 = [uavPos(:, u); params.uavHeight];
            dist = norm(uav3 - ugv3);
            pathLoss = sqrt(params.g0 / dist^params.pathLossExpUGV);
            nlos = (randn + 1i * randn) / sqrt(2);
            h(g, u) = pathLoss * (sqrt(params.kappaUGV / (params.kappaUGV + 1)) + ...
                sqrt(1 / (params.kappaUGV + 1)) * nlos);
        end
    end
end


function assignments = enumerate_assignments(G, U)
    count = G^U;
    assignments = zeros(count, U);
    for idx = 0:(count - 1)
        value = idx;
        for u = 1:U
            assignments(idx + 1, u) = mod(value, G) + 1;
            value = floor(value / G);
        end
    end
end


function out = blend_vector(oldVal, newVal, weight)
    out = (1 - weight) * oldVal + weight * newVal;
end


function out = blend_matrix(oldVal, newVal, weight)
    out = (1 - weight) * oldVal + weight * newVal;
    out = 0.5 * (out + out');
end


function ratio = channel_gain_ratio(params, absH2)
    ratio = absH2 / params.sigmaUAV2;
end


function ok = is_feasible_metrics(params, metrics, W, beta, gamma, p, alphaCCS, alphaG, alphaU)
    ok = alphaCCS >= 1 && alphaCCS <= params.Kccs;
    ok = ok && all(alphaG >= 1) && all(alphaG <= params.Kg);
    ok = ok && all(alphaU >= 1) && all(alphaU <= params.Ku);
    ok = ok && all(sum(beta, 1) == 1);
    ok = ok && all(sum(beta, 2) <= params.betaMax);
    ok = ok && all(gamma >= 1e-3 - 1e-8) && sum(gamma) <= 1 + 1e-8;
    ok = ok && all(p >= -1e-8) && all(p <= params.PmaxUGV + 1e-8);
    ok = ok && real(trace(W)) <= params.PmaxCCS + 1e-6;
    ok = ok && all(metrics.ccsTx <= params.tth + 1e-5);
    ok = ok && all(metrics.totalDelay <= params.tau + 1e-5);
end


function validate_interval_solution(params, state, n, alphaCCS, alphaG, alphaU, W, beta, gamma, p)
    metrics = build_metrics(params, state, n, alphaCCS, alphaG, alphaU, beta, gamma, p, W);
    assert(is_feasible_metrics(params, metrics, W, beta, gamma, p, alphaCCS, alphaG, alphaU), ...
        'The final interval solution violates at least one paper constraint.');
end


function print_summary(results, params)
    fprintf('\nAverage energy over %d interval(s): %.4f J\n', params.N, results.averageEnergy);
    for n = 1:params.N
        sol = results.intervals{n};
        m = sol.metrics;
        fprintf('\nInterval %d\n', n);
        fprintf('  alpha_CCS      = %d\n', sol.alphaCCS);
        fprintf('  alpha_UGV      = [%s]\n', format_vec(sol.alphaUGV));
        fprintf('  alpha_UAV      = [%s]\n', format_vec(sol.alphaUAV));
        fprintf('  gamma          = [%s]\n', format_vec(sol.gamma));
        fprintf('  p_UGV (W)      = [%s]\n', format_vec(sol.p));
        fprintf('  CCS tx latency = [%s] s\n', format_vec(m.ccsTx));
        fprintf('  UGV upd latency= [%s] s\n', format_vec(m.lupdG));
        fprintf('  UAV tx latency = [%s] s\n', format_vec(m.userTx));
        fprintf('  UAV upd latency= [%s] s\n', format_vec(m.uavUpd));
        fprintf('  Total delay    = [%s] s\n', format_vec(m.totalDelay));
        fprintf('  Total energy   = %.4f J\n', sol.totalEnergy);
    end
end


function txt = format_vec(vec)
    txt = strtrim(sprintf('%.4f ', vec(:).'));
end


function plot_ao_convergence(results, params)
    fig = figure('Visible', 'off');
    hold on;
    for n = 1:params.N
        history = results.intervals{n}.history;
        plot(1:numel(history), history, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    end
    grid on;
    xlabel('AO iteration');
    ylabel('Total energy (J)');
    title('AO Convergence');
    legend(compose('Interval %d', 1:params.N), 'Location', 'northeast');
    saveas(fig, params.figName);
    close(fig);
    fprintf('Convergence plot saved to %s\n', params.figName);
end









