function [] = PlotFigure(nr, n_epochs, train, val, x_label, y_label, y_max)
figure(nr)
plot(1 : n_epochs , train, 'r')
hold on
plot(1 : n_epochs , val, 'b')
ylim([0 y_max]);
hold off
xlabel(x_label);
ylabel(y_label);
legend('training', 'validation');
end

