%data = load('embeddings.mat');
%E = data.embeddings;

%mid = E(:, 1);
%fid = E(:, 2);
%genre = E(:, 3:17);
%features = E(:, 18:end);

[coeff,score,~,~,explained,~] = pca(features);

[~, C] = max(genre, [], 2);
scatter3(score(:, 1), score(:, 2), score(:, 3), ones(size(mid)), C);
