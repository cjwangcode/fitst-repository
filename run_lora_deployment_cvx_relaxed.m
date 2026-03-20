function results = run_lora_deployment_cvx_relaxed()
    clc; clear; close all;
    rng(7);

    if exist('cvx_begin', 'file') ~= 2
        error('CVX is not on the MATLAB path.');
    end

    params = default_parameters();
    params.roundAlpha = false;
    params.figName = 'ao_convergence_interval_1_relaxed.png';

    state = build_scenario(params);
    results = solve_all_intervals(params, state);
    print_summary(results, params);
    figure
    hold on;
    for n = 1:params.N
        history = results.intervals{n}.history;
        plot(1:numel(history), history, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    end
    grid on;
    xlabel('AO iteration');
    ylabel('Total energy (J)');
    title('AO Iteration vs Total Energy');
    legend(compose('Interval %d', 1:params.N), 'Location', 'northeast');
    % plot_ao_convergence(results, params);
end


function params = default_parameters()
    params.N = 1;
    params.G = 2;
    params.U = 3;
    params.M = 4;

    params.tau = 10.0;
    params.tth = 1.2;
    params.B = 15e6;

    params.K1 = 128;
    params.K2 = 32;
    params.fCCS = 100e9;
    params.fUGV = 10e9;
    params.muCCS = 1.0;
    params.muUGV = 1.0;
    params.xiCCS = 0.03;
    params.xiUGV = 0.02;
    params.pCCScomp = 10.0;
    params.pUGVcomp = 3.5;

    params.PmaxCCS = 70.0;
    params.PmaxUGV = 10.0;
    params.betaMax = 2;

    params.dCCS = 100e6 * ones(params.N, 1);
    params.dUGV = 20e6 * ones(params.N, 1);
    params.Fup = 140e9 * ones(params.N, 1);
    params.FferCCS = 100e9 * ones(params.N, 1);
    params.FferUGV = 60e9 * ones(params.N, 1);

    params.g0 = 1.0;
    params.pathLossExpCCS = 2.2;
    params.pathLossExpUGV = 2.6;
    params.kappaCCS = 5.0;
    params.kappaUGV = 1.5;
    params.lambda = 0.1;
    params.antSpacing = params.lambda / 2;
    params.sigmaCCS2 = 2e-7;
    params.sigmaUAV2 = 1e-5;

    params.ccsPos = [0.0; 0.0; 60.0];
    params.uavHeight = 60.0;

    params.outerMaxIter = 8;
    params.outerTol = 1e-4;
    params.s2MaxIter = 20;
    params.s2Tol = 1e-4;
    params.gammaMin = 1e-4;
    params.epsVal = 1e-6;
end

function state = build_scenario(params)
    state.ugvPos = zeros(2, params.G, params.N);
    state.uavPos = zeros(2, params.U, params.N);

    ugvBase = [-30.0, 30.0;
               -18.0, 18.0];
    ugvVel = [2.5, -1.8;
              0.9,  0.6];

    uavBase = [-45.0,  5.0, 55.0;
               30.0, 42.0, 22.0];
    uavVel = [ 1.5, -0.7, -1.1;
               0.6,  0.8, -0.5];

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
    results.intervals = cell(params.N, 1);
    totalEnergy = 0.0;

    for n = 1:params.N
        intervalState.hCCS = state.hCCS{n};
        intervalState.hUGV = state.hUGV{n};
        results.intervals{n} = solve_single_interval(params, intervalState, n);
        totalEnergy = totalEnergy + results.intervals{n}.totalEnergy;
    end

    results.averageEnergy = totalEnergy / params.N;
end

function sol = solve_single_interval(params, state, n)
    G = params.G;
    M = params.M;

    W = (0.7 * params.PmaxCCS / M) * eye(M);
    alphaUGV = 12.0 * ones(G, 1);
    beta = initial_association(params);
    gamma = ones(G, 1) / G;
    p = params.PmaxUGV * ones(G, 1);

    prevObjective = inf;
    history = zeros(params.outerMaxIter, 1);

    for iter = 1:params.outerMaxIter
        ccsLat = compute_ccs_tx_latency(params, state.hCCS, W, n);
        ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
        tUGV = ugvTx.groupMaxLatency;

        alphaCCS = solve_s1_alpha_ccs(params, alphaUGV, ccsLat, tUGV, n);
        lupdCCS = ccs_update_latency(params, alphaCCS, n);

        W = solve_s2_beamforming(params, state.hCCS, alphaCCS, alphaUGV, tUGV, n);
        ccsLat = compute_ccs_tx_latency(params, state.hCCS, W, n);

        alphaUGV = solve_s3_alpha_ugv(params, alphaCCS, ccsLat, tUGV, n);

        beta = solve_s4_association(params, state.hUGV, gamma, p);
        ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
        tUGV = ugvTx.groupMaxLatency;

        gamma = solve_s5_bandwidth(params, state.hUGV, beta, p, alphaCCS, alphaUGV, ccsLat, n);
        ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
        tUGV = ugvTx.groupMaxLatency;

        p = solve_s6_power(params, state.hUGV, beta, gamma, alphaCCS, alphaUGV, ccsLat, n);
        ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
        tUGV = ugvTx.groupMaxLatency;

        [objective, ~, ~] = total_interval_energy(params, alphaCCS, alphaUGV, W, p, ccsLat, tUGV, n);
        history(iter) = objective;

        if prevObjective < inf
            relGap = abs(prevObjective - objective) / max(1.0, abs(prevObjective));
            if relGap <= params.outerTol
                history = history(1:iter);
                break;
            end
        end
        prevObjective = objective;
    end

    relaxedAlphaCCS = alphaCCS;
    relaxedAlphaUGV = alphaUGV;
    relaxedObjective = history(end);

    if params.roundAlpha
        alphaCCS = recover_integer_alpha(alphaCCS, params.K1);
        alphaUGV = arrayfun(@(x) recover_integer_alpha(x, params.K2), alphaUGV);
    end

    lupdCCS = ccs_update_latency(params, alphaCCS, n);
    lupdUGV = ugv_update_latency(params, alphaUGV, n);
    ccsLat = compute_ccs_tx_latency(params, state.hCCS, W, n);
    ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
    tUGV = ugvTx.groupMaxLatency;
    totalDelayPerUGV = lupdCCS + ccsLat(:) + lupdUGV(:) + tUGV(:);

    [objective, eCCS, eUGV] = total_interval_energy(params, alphaCCS, alphaUGV, W, p, ccsLat, tUGV, n);
    validate_interval_solution(params, state, alphaCCS, alphaUGV, W, beta, gamma, p, n);

    sol.alphaCCS = alphaCCS;
    sol.alphaUGV = alphaUGV;
    sol.relaxedAlphaCCS = relaxedAlphaCCS;
    sol.relaxedAlphaUGV = relaxedAlphaUGV;
    sol.W = W;
    sol.beta = beta;
    sol.gamma = gamma;
    sol.p = p;
    sol.ccsLatency = ccsLat;
    sol.ugvTx = ugvTx;
    sol.lupdCCS = lupdCCS;
    sol.lupdUGV = lupdUGV;
    sol.totalDelayPerUGV = totalDelayPerUGV;
    sol.ECCS = eCCS;
    sol.EUGV = eUGV;
    sol.totalEnergy = objective;
    sol.history = history;
    sol.relaxedObjective = relaxedObjective;
    sol.finalPointEnergy = objective;
end

function alphaCCS = solve_s1_alpha_ccs(params, alphaUGV, ccsLat, tUGV, n)
    numerCCS = (params.Fup(n) + params.FferCCS(n)) * params.muCCS / params.fCCS;
    numerUGV = params.FferUGV(n) * params.muUGV / params.fUGV;
    lowerBounds = zeros(params.G, 1);
    for g = 1:params.G
        [ccsBudget, ~] = split_update_budget(params, ccsLat, tUGV, numerCCS, numerUGV, n, g);
        lowerBounds(g) = numerCCS / ccsBudget;
    end
    alphaCCS = min(params.K1, max(lowerBounds));
    alphaCCS = max(alphaCCS, params.G + 1.0);
end


function W = solve_s2_beamforming(params, hCCS, alphaCCS, alphaUGV, tUGV, n)
    G = params.G;
    M = params.M;
    Hg = cell(G, 1);
    for g = 1:G
        hg = hCCS(:, g);
        Hg{g} = hg * hg';
    end

    delayBudget = zeros(G, 1);
    for g = 1:G
        delayBudget(g) = params.tau - ccs_update_latency(params, alphaCCS, n) - ...
            ugv_update_latency(params, alphaUGV(g), n) - tUGV(g);
    end
    etaUpper = min([params.tth; delayBudget]);
    if etaUpper <= 0
        error('S2 infeasible at interval %d because the remaining delay budget is non-positive.', n);
    end

    etaLower = min(0.20, 0.5 * etaUpper);
    etaGrid = linspace(etaLower, etaUpper, 16);
    bestEnergy = inf;
    bestW = [];

    for k = 1:numel(etaGrid)
        eta = etaGrid(k);
        if eta <= 0
            continue;
        end
        snrReq = params.sigmaCCS2 * (2^(params.dCCS(n) / (params.B * eta)) - 1);
        cvx_begin quiet sdp
            variable Wvar(M, M) hermitian semidefinite
            minimize(real(trace(Wvar)))
            subject to
                real(trace(Wvar)) <= params.PmaxCCS;
                for g = 1:G
                    real(trace(Hg{g} * Wvar)) >= snrReq;
                end
        cvx_end

        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            energyCand = eta * real(trace(Wvar));
            if energyCand < bestEnergy
                bestEnergy = energyCand;
                bestW = Wvar;
            end
        end
    end

    if isempty(bestW)
        error('S2 beamforming failed at interval %d: no feasible eta in the search grid.', n);
    end
    W = bestW;
end

function alphaUGV = solve_s3_alpha_ugv(params, alphaCCS, ccsLat, tUGV, n)
    numerCCS = (params.Fup(n) + params.FferCCS(n)) * params.muCCS / params.fCCS;
    numerUGV = params.FferUGV(n) * params.muUGV / params.fUGV;
    lowerBounds = zeros(params.G, 1);
    for g = 1:params.G
        [~, ugvBudget] = split_update_budget(params, ccsLat, tUGV, numerCCS, numerUGV, n, g);
        lowerBounds(g) = numerUGV / ugvBudget;
    end
    alphaCommon = min(params.K2, max(lowerBounds));
    alphaCommon = max(alphaCommon, 1.0);
    alphaUGV = alphaCommon * ones(params.G, 1);
end

function [ccsBudget, ugvBudget] = split_update_budget(params, ccsLat, tUGV, numerCCS, numerUGV, n, g)
    sharedBudget = params.tau - ccsLat(g) - tUGV(g) - params.xiCCS - params.xiUGV;
    if sharedBudget <= 0
        error('Update-budget split infeasible at interval %d for UGV %d.', n, g);
    end
    totalNumer = numerCCS + numerUGV;
    ccsBudget = sharedBudget * numerCCS / totalNumer;
    ugvBudget = sharedBudget * numerUGV / totalNumer;
end


function beta = solve_s4_association(params, hUGV, gamma, p)
    G = params.G;
    U = params.U;
    rates = zeros(G, U);
    for g = 1:G
        for u = 1:U
            rates(g, u) = ugv_uav_rate(params, abs(hUGV(g, u))^2, gamma(g), p(g));
        end
    end
    beta = heuristic_association(params, rates);
end

function beta = heuristic_association(params, rates)
    G = params.G;
    U = params.U;
    beta = zeros(G, U);
    load = zeros(G, 1);
    assigned = false(1, U);

    [~, order] = sort(max(rates, [], 1), 'ascend');
    for idx = 1:U
        u = order(idx);
        bestG = 0;
        bestMinRate = -inf;
        bestOwnRate = -inf;
        for g = 1:G
            if load(g) >= params.betaMax
                continue;
            end
            betaCand = beta;
            betaCand(g, u) = 1;
            userRates = sum(betaCand .* rates, 1);
            assignedCand = assigned;
            assignedCand(u) = true;
            minRateCand = min(userRates(assignedCand));
            if minRateCand > bestMinRate + 1e-9 || ...
               (abs(minRateCand - bestMinRate) <= 1e-9 && rates(g, u) > bestOwnRate)
                bestG = g;
                bestMinRate = minRateCand;
                bestOwnRate = rates(g, u);
            end
        end
        if bestG == 0
            error('Heuristic S4 failed: no feasible UGV for UAV %d.', u);
        end
        beta(bestG, u) = 1;
        load(bestG) = load(bestG) + 1;
        assigned(u) = true;
    end

    improved = true;
    while improved
        improved = false;
        currentMinRate = min(sum(beta .* rates, 1));
        for u = 1:U
            g0 = find(beta(:, u) > 0.5, 1, 'first');
            for g = 1:G
                if g == g0 || load(g) >= params.betaMax
                    continue;
                end
                betaCand = beta;
                betaCand(g0, u) = 0;
                betaCand(g, u) = 1;
                minRateCand = min(sum(betaCand .* rates, 1));
                if minRateCand > currentMinRate + 1e-9
                    beta = betaCand;
                    load(g0) = load(g0) - 1;
                    load(g) = load(g) + 1;
                    currentMinRate = minRateCand;
                    improved = true;
                end
            end
        end
    end
end

function gamma = solve_s5_bandwidth(params, hUGV, beta, p, alphaCCS, alphaUGV, ccsLat, n)
    G = params.G;
    U = params.U;
    coeff = zeros(G, U);
    k = zeros(G, 1);
    budget = zeros(G, 1);
    for g = 1:G
        for u = 1:U
            coeff(g, u) = params.B * log2(1 + channel_gain_ratio(params, abs(hUGV(g, u))^2) * p(g));
        end
    end

    lupdCCS = ccs_update_latency(params, alphaCCS, n);
    lupdUGV = ugv_update_latency(params, alphaUGV, n);
    for g = 1:G
        users = find(beta(g, :) > 0.5);
        budget(g) = params.tau - lupdCCS - ccsLat(g) - lupdUGV(g);
        if budget(g) <= 0
            error('S5 infeasible at interval %d for UGV %d due to negative delay budget.', n, g);
        end
        if isempty(users)
            k(g) = 0.0;
        else
            k(g) = max(params.dUGV(n) ./ coeff(g, users).');
        end
    end

    gammaLower = max(params.gammaMin * ones(G, 1), k ./ budget);
    if sum(gammaLower) > 1 + 1e-8
        error('S5 infeasible at interval %d because the minimum feasible bandwidth ratios exceed 1.', n);
    end

    cvx_begin quiet
        variable gammaVar(G)
        minimize(sum((p .* k) .* inv_pos(gammaVar)))
        subject to
            gammaVar >= gammaLower;
            gammaVar <= 1;
            sum(gammaVar) <= 1;
    cvx_end

    if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
        error('S5 bandwidth allocation failed at interval %d: %s', n, cvx_status);
    end
    gamma = gammaVar;
end

function p = solve_s6_power(params, hUGV, beta, gamma, alphaCCS, alphaUGV, ccsLat, n)
    G = params.G;
    p = zeros(G, 1);
    lupdCCS = ccs_update_latency(params, alphaCCS, n);
    lupdUGV = ugv_update_latency(params, alphaUGV, n);

    for g = 1:G
        users = find(beta(g, :) > 0.5);
        if isempty(users)
            p(g) = 0.0;
            continue;
        end

        budget = params.tau - lupdCCS - ccsLat(g) - lupdUGV(g);
        if budget <= 0
            error('S6 infeasible at interval %d for UGV %d due to non-positive delay budget.', n, g);
        end

        aVals = zeros(numel(users), 1);
        for k = 1:numel(users)
            u = users(k);
            aVals(k) = channel_gain_ratio(params, abs(hUGV(g, u))^2);
        end
        aWorst = min(aVals);
        reqSnr = 2^(params.dUGV(n) / (gamma(g) * params.B * budget)) - 1;
        pReq = reqSnr / aWorst;
        if pReq > params.PmaxUGV + 1e-8
            error('S6 infeasible at interval %d for UGV %d because the required power exceeds Pmax.', n, g);
        end
        p(g) = max(params.epsVal, min(params.PmaxUGV, pReq));
    end
end

function alphaInt = recover_integer_alpha(alphaRelaxed, alphaMax)
    alphaInt = min(alphaMax, max(1, ceil(alphaRelaxed)));
end

function lat = compute_ccs_tx_latency(params, hCCS, W, n)
    G = params.G;
    lat = zeros(G, 1);
    for g = 1:G
        hg = hCCS(:, g);
        rate = params.B * log2(1 + real(hg' * W * hg) / params.sigmaCCS2);
        lat(g) = params.dCCS(n) / rate;
    end
end

function tx = compute_uav_tx_latency(params, hUGV, beta, gamma, p, n)
    G = params.G;
    U = params.U;
    userRate = zeros(U, 1);
    for u = 1:U
        for g = 1:G
            userRate(u) = userRate(u) + beta(g, u) * ugv_uav_rate(params, abs(hUGV(g, u))^2, gamma(g), p(g));
        end
    end
    userLatency = params.dUGV(n) ./ userRate;
    groupMaxLatency = zeros(G, 1);
    for g = 1:G
        users = find(beta(g, :) > 0.5);
        if isempty(users)
            groupMaxLatency(g) = 0;
        else
            groupMaxLatency(g) = max(userLatency(users));
        end
    end
    tx.userRate = userRate;
    tx.userLatency = userLatency;
    tx.groupMaxLatency = groupMaxLatency;
end

function [objective, eCCS, eUGV] = total_interval_energy(params, alphaCCS, alphaUGV, W, p, ccsLat, tUGV, n)
    eCCS = alphaCCS * params.pCCScomp * ccs_update_latency(params, alphaCCS, n) + ...
        real(trace(W)) * max(ccsLat);
    eUGV = zeros(params.G, 1);
    for g = 1:params.G
        eUGV(g) = alphaUGV(g) * params.pUGVcomp * ugv_update_latency(params, alphaUGV(g), n) + ...
            p(g) * tUGV(g);
    end
    objective = eCCS + sum(eUGV);
end

function validate_interval_solution(params, state, alphaCCS, alphaUGV, W, beta, gamma, p, n)
    ccsLat = compute_ccs_tx_latency(params, state.hCCS, W, n);
    ugvTx = compute_uav_tx_latency(params, state.hUGV, beta, gamma, p, n);
    lupdCCS = ccs_update_latency(params, alphaCCS, n);
    lupdUGV = ugv_update_latency(params, alphaUGV, n);

    assert(alphaCCS >= -1e-8 && alphaCCS <= params.K1 + 1e-8);
    assert(all(alphaUGV >= -1e-8) && all(alphaUGV <= params.K2 + 1e-8));
    assert(abs(sum(gamma) - 1) <= 1e-3 || sum(gamma) < 1);
    assert(all(gamma >= params.gammaMin - 1e-8));
    assert(all(sum(beta, 1) == 1));
    assert(all(sum(beta, 2) <= params.betaMax));
    assert(all(p >= -1e-8) && all(p <= params.PmaxUGV + 1e-8));
    assert(real(trace(W)) <= params.PmaxCCS + 1e-6);
    assert(all(ccsLat <= params.tth + 1e-4));
    for g = 1:params.G
        totalDelay = lupdCCS + ccsLat(g) + lupdUGV(g) + ugvTx.groupMaxLatency(g);
        assert(totalDelay <= params.tau + 1e-4, 'Delay violation at interval %d, UGV %d.', n, g);
    end
end

function val = ccs_update_latency(params, alphaCCS, n)
    val = (params.Fup(n) + params.FferCCS(n)) * params.muCCS / (max(alphaCCS, params.epsVal) * params.fCCS) + params.xiCCS;
end

function val = ugv_update_latency(params, alphaUGV, n)
    val = params.FferUGV(n) * params.muUGV ./ (max(alphaUGV, params.epsVal) * params.fUGV) + params.xiUGV;
end

function rate = ugv_uav_rate(params, absH2, gamma, p)
    rate = gamma * params.B * log2(1 + channel_gain_ratio(params, absH2) * p);
end

function a = channel_gain_ratio(params, absH2)
    a = absH2 / params.sigmaUAV2;
end

function beta = initial_association(params)
    beta = zeros(params.G, params.U);
    for u = 1:params.U
        g = mod(u - 1, params.G) + 1;
        beta(g, u) = 1;
    end
end

function h = generate_ccs_ugv_channels(params, ugvPos)
    h = zeros(params.M, params.G);
    for g = 1:params.G
        pos = [ugvPos(:, g); 0.0];
        diff = pos - params.ccsPos;
        dist = norm(diff);
        pathLoss = sqrt(params.g0 / (dist ^ params.pathLossExpCCS));
        aoD = acos((params.ccsPos(3) - pos(3)) / dist);
        idx = (0:(params.M - 1)).';
        steering = exp(-1i * 2 * pi * params.antSpacing / params.lambda * idx * cos(aoD));
        steering = steering / sqrt(params.M);
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
            pathLoss = sqrt(params.g0 / (dist ^ params.pathLossExpUGV));
            nlos = (randn + 1i * randn) / sqrt(2);
            h(g, u) = pathLoss * (sqrt(params.kappaUGV / (params.kappaUGV + 1)) + ...
                                  sqrt(1 / (params.kappaUGV + 1)) * nlos);
        end
    end
end

function print_summary(results, params)
    fprintf('\nAverage energy over %d intervals: %.6f J\n', params.N, results.averageEnergy);
    for n = 1:params.N
        sol = results.intervals{n};
        fprintf('\nInterval %d\n', n);
        fprintf('  alpha_CCS = %.4f\n', sol.alphaCCS);
        fprintf('  alpha_UGV = [%s]\n', format_vec(sol.alphaUGV));
        if params.roundAlpha
            fprintf('  relaxed alpha_CCS = %.4f\n', sol.relaxedAlphaCCS);
            fprintf('  relaxed alpha_UGV = [%s]\n', format_vec(sol.relaxedAlphaUGV));
        end
        fprintf('  gamma     = [%s]\n', format_vec(sol.gamma));
        fprintf('  p_UGV(W)  = [%s]\n', format_vec(sol.p));
        fprintf('  CCS update latency(s) = %.4f\n', sol.lupdCCS);
        fprintf('  CCS->UGV latency(s)   = [%s]\n', format_vec(sol.ccsLatency));
        fprintf('  UGV update latency(s) = [%s]\n', format_vec(sol.lupdUGV));
        fprintf('  UAV link latency(s)   = [%s]\n', format_vec(sol.ugvTx.userLatency));
        fprintf('  UGV max UAV latency(s)= [%s]\n', format_vec(sol.ugvTx.groupMaxLatency));
        fprintf('  Total delay per UGV(s)= [%s]\n', format_vec(sol.totalDelayPerUGV));
        fprintf('  AO relaxed final energy = %.6f\n', sol.relaxedObjective);
        if params.roundAlpha
            fprintf('  rounded integer energy = %.6f\n', sol.finalPointEnergy);
        end
        fprintf('  total energy = %.6f J\n', sol.totalEnergy);
    end
end

function out = format_vec(vec)
    out = strtrim(sprintf('%.4f ', vec(:).'));
end

function plot_ao_convergence(results, params)
    fig = figure('Visible', 'off');
    hold on;
    legends = cell(1, 2 * params.N);
    legendCount = 0;
    for n = 1:params.N
        history = results.intervals{n}.history;
        plot(1:numel(history), history, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
        legendCount = legendCount + 1;
        legends{legendCount} = sprintf('Interval %d AO', n);
        if params.roundAlpha
            plot(numel(history), results.intervals{n}.finalPointEnergy, 'r*', 'MarkerSize', 9, 'LineWidth', 1.5);
            legendCount = legendCount + 1;
            legends{legendCount} = sprintf('Interval %d rounded', n);
        end
    end
    grid on;
    xlabel('AO iteration');
    ylabel('Total energy (J)');
    title('AO Iteration vs Total Energy');
    legend(legends(1:legendCount), 'Location', 'northeast');
    saveas(fig, params.figName);
    close(fig);
    fprintf('  convergence plot saved to %s\n', params.figName);
end
