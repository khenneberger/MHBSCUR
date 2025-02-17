function fig = plot_method_improved(Methods, x, classifier_names, classifier_id, method_names, dataset_id, type_accu)
    
    M_cnt = size(Methods, 2);
    colors = [linspecer(M_cnt); 0, 0, 0];
    colors_cnt = size(colors, 1);
    line_width = 1.5;
    marker_size = 6;
    line_types = { '-*g','-^b','-om','-vk', '-*g'};
    line_types_cnt = size(line_types, 2);
    x_cnt = size(x, 2);
    y = zeros(1, x_cnt);
    set(gcf,'unit','normalized','position',[0, 0, 0.22,0.35]);
    fig = figure(1);

    ymin = zeros(M_cnt, 1);
    y_max = 0;
    for i = 1 : M_cnt
        for j = 1 : x_cnt
            switch type_accu
                case 1
                    y(1, j) = mean(Methods{1, i}.accu(dataset_id, classifier_id, floor(x(j)/x(1,1)), :));
            end
        end
        ymin(i) = y(1, 1);
        y_max = max(y_max, max(y));
        plot(x, y(1:x_cnt), line_types{1, mod(i, line_types_cnt)+1}, 'LineWidth' , line_width, 'MarkerSize', marker_size);%,'Color', colors(mod(i, colors_cnt)+1, :));
        hold on;
    end
    axis normal;
    xticks([0 6 12 18 24 30]);
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','fontsize',13)
    yticks([0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]);
    xlabel('Number of Bands','FontSize', 14);
    ylabel(['Overall Accuracy by ', classifier_names{classifier_id}],'FontSize', 14);
    grid on;
    h = legend(method_names, 'Location', 'southeast');
    exportgraphics(fig,sprintf([classifier_names{classifier_id},'_accuracy_%d.png'],dataset_id))
    hold off
end
