clear;
gmax = 5; % S/cm^2
onset = 15; % ms
t = 0:0.05:100; % ms
tau = 3.0; % ms

gsyn = gmax * (t>=onset) .*(t-onset)/tau .* exp(-(t-onset-tau)/tau);

gsyn(gsyn<1e-9) = 1e-9;
rsyn = 1./gsyn;
rsyn1 = [t',rsyn'];

% plot(t, rsyn);
% ylim([0,1]);