function res = voc_eval(path, comp_id, test_set, output_dir, rm_res)

VOCopts = get_voc_opts(path);
VOCopts.testset = test_set;

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
fprintf('average precision:\n')
for i = 1:length(VOCopts.classes)
  cls = VOCopts.classes{i};
  res(i) = voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res);
  fprintf('%d_%s : %.2f %.2f\n', i, cls, res(i).ap*100, res(i).ap_auc*100);
end

aps = [res(:).ap]';
%fprintf('%.1f\n', aps * 100);
fprintf('mean:%.1f\n', mean(aps) * 100);
fprintf('~~~~~~~~~~~~~~~~~~~~\n');

function res = voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res)

test_set = VOCopts.testset;
year = VOCopts.dataset(4:end);

addpath(fullfile(VOCopts.datadir, 'VOCcode'));
%sprintf('VOCopts.datadir:%s', VOCopts.datadir)

res_fn = sprintf(VOCopts.detrespath, comp_id, cls);

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

do_eval = (str2num(year) <= 2007) | ~strcmp(test_set, 'test');
if do_eval
  % Bug in VOCevaldet requires that tic has been called first
  tic;
  [recall, prec, ap] = VOCevaldet(VOCopts, comp_id, cls, true);
  ap_auc = xVOCap(recall, prec);

  % force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', ...
        [output_dir '/' cls '_pr.jpg']);
end
%fprintf('average precision %s : %.2f %.2f\n', cls, ap*100, ap_auc*100);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

% plotROC
filename = sprintf('%s/results/VOC2007/Main/roc_%s', VOCopts.datadir, cls);
%fprintf('filename:%s\n',filename);
plot_roc(filename,recall,prec);

save([output_dir '/' cls '_pr.mat'], ...
     'res', 'recall', 'prec', 'ap', 'ap_auc');

if rm_res
  delete(res_fn);
end

rmpath(fullfile(VOCopts.datadir, 'VOCcode'));

%% plotROC
function plot_roc(filename,recall,precision)

plot(recall,precision,'r','LineWidth',1);
% hold on;
axis([0 1 0 1]);
title('ROC');
xlabel('Recall');
ylabel('Precision');
grid on;
saveas(gcf,filename,'jpg');

