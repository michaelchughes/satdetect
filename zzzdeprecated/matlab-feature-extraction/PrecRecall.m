%FN:  0/15   FP:  9/1000   acc:1006/1015 0.991   
%FN:  1/15   FP:  3/1000   acc:1011/1015 0.996   
%FN:  2/15   FP:  0/1000   acc:1013/1015 0.998   
%FN:  3/15   FP:  0/1000   acc:1012/1015 0.997 

% prec = [15/(15+9) 14/(14+3) 13/13];
% recall = [15/15 14/15 13/15];
% 
% plot( recall, prec, 'ko-')


tpr = [1 14/15 13/15];
fpr = [9/(15+9)  3/(14+3) 0];
figure(1); clf; hold('on');

plot( fpr, tpr, 'ro-', 'LineWidth', 2);
chance = linspace(0, 1, 50);
plot( chance, chance, '--', 'LineWidth', 2);
ylim([0, 1]);
xlim([0, 1]);
ylabel('true pos rate');
xlabel('false pos rate');
legend('our classifier', 'chance');
set(findall(gcf,'type','text'),'FontSize', 25);
set(findall(gcf,'type','axes'),'FontSize', 25);