%% 
clear all; close all; clc;

%% ARX
for i = 3:4
    disp(['--------------------- Dados ' num2str(i) ' --------------------- ' ])
    filename = ['dados_' num2str(i) '.txt'];
    dados{i} = load(filename);   
    dado = cell2mat(dados(1,i));

    u = dado(:,2); %Dados de entrada do sistema
    y = dado(:,1); %Dados de saída do sistema
    N = length(y); %Quantidade de observações

    lambda = 0.97;
    
    MSEs = [];
    COEFFs = [];

    for ordem = 1:5 %Ordem do sistema
        phi = montaMatReg(ordem, y, u, true, 0); % Matriz de regressores

        P = eye(ordem*2)*10000;
        theta = rand(ordem*2,1);
        thetas = theta;
        
        for k = 1:N-ordem
            K = P*phi(k,:)'/(phi(k,:)*P*phi(k,:)'+lambda);
            theta = theta + K*(y(k+ordem) - phi(k,:)*theta);
            P = (1/lambda) * (P - (P*phi(k,:)'*phi(k,:)*P)/(phi(k,:)*P*phi(k,:)'+lambda));
            thetas = [thetas theta];
            
            erro = y(ordem+1:end) - phi*theta;
            y_mean = mean(phi*theta);
            
            MSE = sum(erro.^2);
            COEFF = 1 - sum(erro.^2)/sum((y - y_mean).^2);           
        end
        
        MSEs = [MSEs MSE];
        COEFFs = [COEFFs COEFF];
        
        figure
        for graph = 1:ordem
            subplot(ordem, 2, 2*graph-1)
            plot(thetas(graph,:))
            ylabel(['\theta' num2str(graph)])
            
            subplot(ordem, 2, 2*graph)
            plot(thetas(graph + ordem,:))
            ylabel(['\theta' num2str(graph + ordem)])
            
            suptitle(['ARX - Conjunto de dados ' num2str(i) ' - Ordem: ' num2str(ordem)])
        end
    end
    
    disp('ARX');
    disp(MSEs);
    disp(COEFFs);
end

%% ARMAX

for i = 3:4
    disp(['--------------------- Dados ' num2str(i) ' --------------------- ' ])
    filename = ['dados_' num2str(i) '.txt'];
    dados{i} = load(filename);   
    dado = cell2mat(dados(1,i));

    u = dado(:,2); %Dados de entrada do sistema
    y = dado(:,1); %Dados de saída do sistema
    N = length(y); %Quantidade de observações

    lambda = 0.97;
    
    MSEs = [];
    COEFFs = [];

    for ordem = 1:5 %Ordem do sistema
        e = (0 + (0.05).*rand(N,1));
        
        for it = 1: 50
            phi = montaMatReg(ordem, y, u, false, e); % Matriz de regressores

            P = eye(ordem*3)*10000;
            theta = rand(ordem*3,1);
            thetas = theta;

            for k = 1:N-ordem
                K = P*phi(k,:)'/(phi(k,:)*P*phi(k,:)'+lambda);
                theta = theta + K*(y(k+ordem) - phi(k,:)*theta);
                P = (1/lambda) * (P - (P*phi(k,:)'*phi(k,:)*P)/(phi(k,:)*P*phi(k,:)'+lambda));
                thetas = [thetas theta];

                erro = y(ordem+1:end) - phi*theta;
                y_mean = mean(phi*theta);

                MSE = sum(erro.^2);
                COEFF = 1 - sum(erro.^2)/sum((y - y_mean).^2);           
            end
        end
        
        MSEs = [MSEs MSE];
        COEFFs = [COEFFs COEFF];
        
        figure
        for graph = 1:ordem
            subplot(ordem, 2, 2*graph-1)
            plot(thetas(graph,:))
            ylabel(['\theta' num2str(graph)])
            
            subplot(ordem, 2, 2*graph)
            plot(thetas(graph + ordem,:))
            ylabel(['\theta' num2str(graph + ordem)])
            
            suptitle(['ARMAX - Conjunto de dados ' num2str(i) ' - Ordem: ' num2str(ordem)])
        end
    end
    
    disp('ARMAX');
    disp(MSEs);
    disp(COEFFs);
end


