classdef DPTSVM < handle
% DPTSVM — Double-Side Projection Twin SVM 
% Using MM to optimize the primal objective with L1-norm within-class term and double-sided hinge loss.
% Usage:
%   Param.c1 = 1;        % the parameter for L1-norm within-class term
%   Param.c2 = 1;        %  the parameter for double-sided hinge loss
%   Param.solver = "SOR";  %  solver: "QP" or "SOR" or "ADMM" (default "ADMM")


  properties
    c1; c2;
    Ker;

    nclass;
    w;    % cell array storing two projection vectors
    m_k;    % class centers
    solver;  % default solver
  end

  methods
    function obj = DPTSVM(param)
      if nargin < 1, param = struct(); end
      obj.c1 = getfield_def(param, 'c1', 1);
      obj.c2 = getfield_def(param, 'c2', 1);
      obj.solver = getfield_def(param, 'solver', "ADMM");


    end

    function [obj,Times] = train(obj,TrainData)
      tic;
      X = TrainData.X;
      Y = TrainData.Y;

      labels = unique(Y);
      obj.nclass = length(labels);
      if obj.nclass ~= 2
        error('Only binary DPTSVM is implemented.');
      end

      % Split data into two classes
      A1 = X(Y==labels(1),:);
      A2 = X(Y==labels(2),:);

      % Compute class centers
      m1 = mean(A1,1)';
      m2 = mean(A2,1)';

      % Shift samples around class centers
      Ak1 = A1 - m1';   Bk1 = A2 - m1';
      Ak2 = A2 - m2';   Bk2 = A1 - m2';

      obj.m_k = {m1, m2};
      obj.w = cell(2,1);

      % Solve two projection directions
      if obj.solver == "SOR"
        obj.w{1} = obj.solveOneDirection_SOR(Ak1, Bk1);
        obj.w{2} = obj.solveOneDirection_SOR(Ak2, Bk2);
      elseif obj.solver == "ADMM"
        obj.w{1} = obj.solveOneDirection_ADMM(Ak1, Bk1);
        obj.w{2} = obj.solveOneDirection_ADMM(Ak2, Bk2);
      elseif obj.solver == "QP"
        obj.w{1} = obj.solveOneDirection_QP(Ak1, Bk1);
        obj.w{2} = obj.solveOneDirection_QP(Ak2, Bk2);
      else
        error('Unknown solver: %s', obj.solver);
      end

      Times = toc;
    end

    function Result = test(obj,TestData)
      X = TestData.X;
      n = size(X,1);

      dist = zeros(n,2);
      for k=1:2
        wk = obj.w{k};
        mk = obj.m_k{k};
        Xc = X - mk';
        dist(:,k) = abs(Xc * wk);
      end

      [~,PredictY] = min(dist,[],2);

      Result.PredictY = PredictY;
      Result.d(:,1) = dist(:,1); Result.d(:,2) = dist(:,2);
      % Result.Y        = dist(:,1) - dist(:,2);
      Result.Y        = 1./(1+exp(dist(:,1) - dist(:,2)));
      %Result.Acc      = sum(PredictY == TestData.Y)/n;
    end
  end

  % ============================================================
  %            OPTIMIZED MM + DUAL QP SOLVER FOR ONE CLASS
  % ============================================================
  methods (Access=private)
    function w = solveOneDirection_QP(obj,A,B)
      % A : within-class centered samples (n1 x d)
      % B : opposite-class centered samples (n2 x d)
      %
      % Objective:
      %   min_w 0.5||w||^2 + (c1/2)*||A w||_1 + c2 * sum max(0,1-|B w|)
      %
      % Using MM:
      %   L1 term approximated by reweighted quadratic
      %   hinge(|B w|) approximated using sign matrix F
      %
      % with matlab quadprog to solve the dual QP in each MM iteration.

      [n1,d] = size(A);
      n2 = size(B,1);

      maxIter = 40;
      epsD = 1e-6;
      tol = 1e-3;

      % Random init
      % w = randn(d,1);
      w = ones(d,1);
      Jprev = inf;

      % Precompute A'*A once (costly but reused)
      At = A';
      Bt = B';

      % quadprog options
      qpopt = optimoptions('quadprog',...
          'Display','off',...
          'Algorithm','interior-point-convex');

      for it=1:maxIter

        % -----------------------------
        %  (1) Update D for L1(Aw)
        % -----------------------------
        Aw = A*w;  % n1 x 1
        Dvec = 1./(abs(Aw) + epsD);   % n1 x 1
        % Instead of forming diag(Dvec), use elementwise multiplication later

        % -----------------------------
        %  (2) Build K = I + c1*A'*D*A efficiently
        % -----------------------------
        % Woodbury form:
        %   K = I + c1 * A' * (diag(Dvec)) * A
        %
        % Let U = sqrt(c1)*A' * diag(sqrt(Dvec))
        %     K = I + U * U'
        sqrtD = sqrt(Dvec);
        U = At .* (sqrtD');      % (d x n1) = A' * diag(sqrtD)
        U = sqrt(obj.c1) * U;    % incorporate c1^{1/2}

        % Cholesky: solve (I + U*U') later via L factor
        % Compute Cholesky of I + U*U'
        % This is O(d^2 * n1) but cheaper than full K inv each iter
        Kmat = eye(d) + U*U';
        [Lflag,p] = chol(Kmat,'lower');
        if p>0
            % add jitter for numerical safety
            Kmat = Kmat + 1e-6*eye(d);
            Lflag = chol(Kmat,'lower');
        end
        L = Lflag;   % L * L' = K

        % -----------------------------
        %  (3) Update F for hinge(|B w|)
        % -----------------------------
        Bw = B*w;
        s = sign(Bw);
        s(s==0)=1;
        F = s;   % store only vector; will use elementwise multiply

        % -----------------------------
        %  (4) Build dual QP matrix Q
        %     Q = F * B * K^{-1} * B' * F
        % -----------------------------
        % Compute:
        %   Z = K^{-1} * B'
        % via Cholesky solves
        %   Solve L y = B'  -> y
        %   Solve L' Z = y  -> Z
        Ytemp = L \ Bt;      % solve L * Ytemp = B'
        Z = L' \ Ytemp;      % solve L' * Z = Ytemp

        % Now Q = diag(F) * (B*Z) * diag(F)
        % To avoid forming diag(F): do elementwise multiplication
        BZ = B * Z;          % (n2 x n2)
        Q = (F .* BZ) .* F'; % broadcasting F[i]*BZ[i,j]*F[j]
        % Symmetrize for safety
        Q = 0.5*(Q + Q');

        f = -ones(n2,1);
        lb = zeros(n2,1);
        ub = obj.c2 * ones(n2,1);

        alpha = quadprog(Q,f,[],[],[],[],lb,ub,[],qpopt);

        % -----------------------------
        %  (5) Recover w = K^{-1} * (B' * (F .* alpha))
        % -----------------------------
        rhs = Bt * (F .* alpha);   % d x 1

        y = L \ rhs;
        w_new = L' \ y;

        % -----------------------------
        %  (6) Compute objective for convergence check
        % -----------------------------
        Aw_new = A*w_new;
        Bw_new = B*w_new;

        J = 0.5*(w_new'*w_new) + ...
            (obj.c1/2)*sum(abs(Aw_new)) + ...
            obj.c2*sum(max(0,1-abs(Bw_new)));

        if abs(Jprev - J) < tol
            w = w_new;
            break;
        end

        w = w_new;
        Jprev = J;
      end
    end
  
    function w = solveOneDirection_SOR(obj,A,B)
      % A : within-class centered samples (n1 x d)
      % B : opposite-class centered samples (n2 x d)
      %
      % Objective:
      %   min_w 0.5||w||^2 + (c1/2)*||A w||_1 + c2 * sum_j max(0, 1 - |B w|_j)
      %
      % This method uses MM on the primal and SOR on the dual QP:
      %   min_{0<=alpha<=c2} 0.5 * alpha' Q alpha - alpha' e
      %   Q = F * B * K^{-1} * B' * F,
      % where K = I + c1 * A' * D * A, D diagonal from L1 reweighting.
      % with SOR to solve the dual QP in each MM iteration.

      [n1,d] = size(A);
      n2 = size(B,1);

      % Outer MM iterations (on w, D, F)
      maxIterMM  = 30;   % you can tune this
      epsD       = 1e-5;
      tolMM      = 1e-3;

      % SOR parameters (for dual QP)
      maxIterSOR = 200;  % typical 100–300 is enough
      tolSOR     = 1e-4;
      omega      = 1.6;  % relaxation parameter in (0,2)

      % Initialization of w
      w     = ones(d,1);
      Jprev = inf;

      At = A';
      Bt = B';

      for it = 1:maxIterMM
        % ------------------------------------------------
        % (1) Update D from L1(Aw) for IRLS
        % ------------------------------------------------
        Aw   = A * w;                 % n1 x 1
        Dvec = 1 ./ (abs(Aw) + epsD); % n1 x 1

        % ------------------------------------------------
        % (2) Build K = I + c1 * A' * D * A via Woodbury-like form
        %     Use Cholesky factorization for K^{-1} operations
        % ------------------------------------------------
        sqrtD = sqrt(Dvec);           % n1 x 1
        U = At .* (sqrtD');           % (d x n1) = A' * diag(sqrtD)
        U = sqrt(obj.c1) * U;         % incorporate sqrt(c1)
        Kmat = eye(d) + U * U';       % d x d

        [L,p] = chol(Kmat,'lower');
        if p > 0
            % add small jitter if not SPD numerically
            Kmat = Kmat + 1e-6*eye(d);
            L = chol(Kmat,'lower');
        end

        % ------------------------------------------------
        % (3) Update F from sign(Bw) for double-sided margin
        % ------------------------------------------------
        Bw = B * w;         % n2 x 1
        F  = sign(Bw);      % n2 x 1
        F(F==0) = 1;

        % ------------------------------------------------
        % (4) Build dual Q matrix Q = F * B * K^{-1} * B' * F
        %     but we will solve the dual by SOR instead of quadprog
        % ------------------------------------------------
        % Compute Z = K^{-1} * B'
        Ytemp = L \ Bt;     % solve L * Ytemp = B'
        Z     = L' \ Ytemp; % solve L' * Z = Ytemp

        % BZ = B * Z is n2 x n2
        BZ = B * Z;
        % Apply the F scaling on both sides: Q = diag(F) * BZ * diag(F)
        Q = (F .* BZ) .* F';   % broadcast F(i)*BZ(i,j)*F(j)
        Q = 0.5 * (Q + Q');    % symmetrize for numerical stability

        % ------------------------------------------------
        % (5) Solve dual QP: min 0.5*alpha' Q alpha - alpha' e
        %     s.t. 0 <= alpha <= c2
        %     using SOR (projected Gauss–Seidel)
        % ------------------------------------------------
        alpha = obj.sor_box_qp(Q, obj.c2, maxIterSOR, tolSOR, omega);

        % ------------------------------------------------
        % (6) Recover w = K^{-1} * (B' * (F .* alpha))
        % ------------------------------------------------
        rhs = Bt * (F .* alpha);  % d x 1

        y    = L \ rhs;
        wNew = L' \ y;

        % ------------------------------------------------
        % (7) Evaluate objective J(w) for convergence check
        % ------------------------------------------------
        AwNew = A * wNew;
        BwNew = B * wNew;

        J = 0.5*(wNew' * wNew) + ...
            (obj.c1/2) * sum(abs(AwNew)) + ...
            obj.c2 * sum(max(0, 1 - abs(BwNew)));

        if abs(J - Jprev) < tolMM
            w = wNew;
            break;
        end

        w     = wNew;
        Jprev = J;
      end
    end

    function alpha = sor_box_qp(~, Q, c2, maxIterSOR, tolSOR, omega)
      % SOR-based solver for box-constrained QP:
      %   min 0.5 * alpha' Q alpha - alpha' e
      %   s.t. 0 <= alpha <= c2
      %
      % Input:
      %   Q           : symmetric positive-definite matrix (n x n)
      %   c2          : upper bound for box constraint
      %   maxIterSOR  : maximum number of SOR sweeps
      %   tolSOR      : stopping tolerance in infinity norm
      %   omega       : relaxation parameter (0, 2)
      %
      % Output:
      %   alpha       : approximate minimizer of the dual problem

      n = size(Q,1);
      alpha = zeros(n,1);

      % Precompute diagonal of Q
      diagQ = diag(Q);

      % Basic projected SOR
      for it = 1:maxIterSOR
        alphaOld = alpha;

        for i = 1:n
          % Gradient component:
          %   g_i = (Q(i,:) * alpha) - 1
          % We separate the Q(i,i)*alpha(i) term for the update
          Qi = Q(i,:);                    % 1 x n row
          tmp = Qi * alpha - Qi(i)*alpha(i);

          % Gauss–Seidel + relaxation:
          %   alpha_i^{new} = (1 - omega)*alpha_i + omega*(1 - tmp) / Q_ii
          a_i = (1 - omega)*alpha(i) + omega*(1 - tmp)/diagQ(i);

          % Project onto [0, c2]
          if a_i < 0
            a_i = 0;
          elseif a_i > c2
            a_i = c2;
          end

          alpha(i) = a_i;
        end

        % Check convergence in infinity norm
        if norm(alpha - alphaOld, inf) < tolSOR
          break;
        end
      end
    end


    function w = solveOneDirection_ADMM(obj, A, B)
      % Fast solver for one projection direction using:
      %   Outer loop: update sign vector F = sign(B*w) (MM-style)
      %   Inner loop: solve the convex subproblem with fixed F via ADMM
      %
      % This avoids constructing the n2-by-n2 dual matrix Q and avoids QP.
      % Complexity is dominated by solving a d-by-d linear system per ADMM step.

      [n1, d] = size(A);
      n2 = size(B, 1);

      % Outer MM parameters
      maxIterMM = 20;        % usually 10~20 is enough
      tolMM     = 1e-3;

      % ADMM parameters (inner)
      maxIterADMM = 100;      % 50~150 typical
      tolADMM     = 1e-4;
      rho1 = 1.0;            % penalty for u = A w
      rho2 = 1.0;            % penalty for v = F B w
      % You can try rho1=rho2 in [0.5, 5] for best speed.

      % Initialization
      w = ones(d,1);
      Jprev = inf;

      AtA = A' * A;          % d-by-d
      BtB = B' * B;          % d-by-d
      I   = eye(d);

      for it = 1:maxIterMM
        % ------------------------------------------------------------
        % (1) Update sign vector F based on current w (MM majorization)
        % ------------------------------------------------------------
        Bw = B * w;
        F  = sign(Bw);
        F(F==0) = 1;

        % Equivalent operator: FB = diag(F) * B, but do it implicitly
        % because diag(F) is never formed.
        % Note: (FB)'(FB) = B'B since diag(F)^2 = I.
        % So the w-update matrix does NOT depend on F (very important!).
        %
        % M = I + rho1*A'*A + rho2*B'*B
        M = I + rho1 * AtA + rho2 * BtB;

        % Pre-factorize once per MM iteration (cheap, d-by-d)
        % If d is small, this is extremely fast.
        [L, p] = chol(M, 'lower');
        if p > 0
            M = M + 1e-8 * I;
            L = chol(M, 'lower');
        end

        % ------------------------------------------------------------
        % (2) ADMM variables: u, v, s and scaled duals y1, y2
        % ------------------------------------------------------------
        u  = A * w;                     % n1 x 1
        v  = (F .* (B * w));            % n2 x 1  (v = F B w)
        s  = max(0, 1 - v);             % n2 x 1  (hinge slack)
        y1 = zeros(n1,1);               % scaled dual for u = A w
        y2 = zeros(n2,1);               % scaled dual for v = F B w

        % ------------------------------------------------------------
        % (3) Inner ADMM iterations
        % ------------------------------------------------------------
        for k = 1:maxIterADMM
          w_old = w;

          % ---- w-update (solve d-by-d system) ----
          % Minimize:
          %   0.5||w||^2 + (rho1/2)||A w - (u - y1)||^2 + (rho2/2)||F(B w) - (v - y2)||^2
          %
          % The normal equation:
          %   (I + rho1 A'A + rho2 B'B) w =
          %       rho1 A' (u - y1) + rho2 B' (F .* (v - y2))
          rhs = rho1 * (A' * (u - y1)) + rho2 * (B' * (F .* (v - y2)));

          % Solve M w = rhs via Cholesky factors
          tmp = L \ rhs;
          w   = L' \ tmp;

          % ---- u-update: proximal operator of (c1/2)*||u||_1 ----
          % u = soft(A w + y1, (c1/2)/rho1)
          z1 = A * w + y1;
          u  = sign(z1) .* max(0, abs(z1) - (obj.c1/2)/rho1);

          % ---- v-update: involves s and linear constraint s = 1 - v, s >= 0 ----
          % We combine v and s updates as:
          %   minimize (rho2/2)||v - (F(Bw)+y2)||^2 + c2*1'*s
          %   s.t. s = 1 - v, s >= 0
          %
          % Substitute s = max(0, 1 - v) and solve for v in closed form:
          % Equivalent prox for hinge: v = argmin (rho2/2)||v - t||^2 + c2*max(0, 1 - v)
          t  = (F .* (B * w)) + y2;   % "t" is the quadratic center

          % Closed-form hinge proximal (elementwise):
          % If v >= 1: penalty 0 -> v = t, but must satisfy v >= 1 -> v = max(t,1)
          % If v < 1: objective (rho2/2)(v-t)^2 + c2(1-v)
          % derivative: rho2(v-t) - c2 = 0 -> v = t + c2/rho2, but must be < 1
          v1 = max(t, 1);
          v2 = min(t + obj.c2/rho2, 1);

          % Choose the better one elementwise by comparing objective values
          obj1 = 0.5*rho2*(v1 - t).^2;                 % hinge part is 0
          obj2 = 0.5*rho2*(v2 - t).^2 + obj.c2*(1-v2); % hinge active
          pick = (obj2 < obj1);
          v    = v1;
          v(pick) = v2(pick);

          % ---- Dual updates (scaled form) ----
          y1 = y1 + (A*w - u);
          y2 = y2 + ((F .* (B*w)) - v);

          % ---- Stopping: small change in w ----
          if norm(w - w_old, 2) / (norm(w_old,2)+1e-12) < tolADMM
            break;
          end
        end

        % ------------------------------------------------------------
        % (4) Evaluate primal objective for MM stopping
        % ------------------------------------------------------------
        Aw = A*w;
        Bw = B*w;
        J = 0.5*(w'*w) + (obj.c1/2)*sum(abs(Aw)) + obj.c2*sum(max(0, 1-abs(Bw)));

        if abs(J - Jprev) < tolMM
          break;
        end
        Jprev = J;
      end
    end


  end
end

%% ========================= utility =========================
function v = getfield_def(s, name, def)
  if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
  else
    v = def;
  end

end
