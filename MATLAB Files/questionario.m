close all
s = tf('s');

MSE_results = containers.Map;

for idx = 1:2
    % Inicio da questão 1
    
    clear A B G Af Bf y Ad Bd u P T
    figure
    sprintf('Sistema %d',idx)
    if idx == 1
        G = (0.5*s^2 + 2*s + 2) / (s^3 + 3*s^2 + 4*s + 2)
    else
        G = 2.5 / (s^2 + s + 2.5)
    end
    
    step(G);
    hold on

    [B, A] = tfdata(G);
    B = cell2mat(B);
    A = cell2mat(A);

    [z,p,k] = tf2zp(B, A)
    G1zp = zpk(z, p, k)

    G1d = c2d(G, 0.1)
    step(G1d);

    [Bd, Ad] = tfdata(G1d, 'v');
    Bf = poly2sym(Bd);
    Af = poly2sym(Ad);

    num = 0;
    den = 0;
    
    syms k u
    
    y(1:100) = 0;
    equation = 'y(k)=';
     for k=length(Ad):100
        for i = 1:length(Ad)-1
           y(k) = y(k)-Ad(i+1)*y(k-i);
        end
        for i = 1:length(Bd)
           y(k) = y(k)+Bd(i)*1;
        end
    end
     hold on
     plot(0:0.1:9.9,y, 'r')
    for i = 2:length(Ad)
        den = den + Ad(i)*(k^(i-1));
        if -Ad(i) > 0 && i ~= 2
            equation = strcat(equation, '+');
        end
        equation = strcat(equation, num2str(-Ad(i)));
        equation = strcat(equation, '*y(k-');
        equation = strcat(equation, num2str(i-1));
        equation = strcat(equation, ')');
    end
    for i = 1:length(Bd)
        if( Bd(i) == 0 )
            continue
        end
        if Bd(i) > 0
            equation = strcat(equation, '+');
        end
        num = num + Bd(i)*(k^(i-1));
        equation = strcat(equation, num2str(Bd(i)));
        equation = strcat(equation, '*u(k-');
        equation = strcat(equation, num2str(i-1));
        equation = strcat(equation, ')');
    end
    
    disp(vpa(equation,3))
    
    % Fim da questão 1
    
    % Inicio da questão 2
    
    N = 100;
    
    u = (-1 + (1+1).*rand(N,1));
    
    y(1:100) = 0;
    for k=length(Ad):100
        for i = 1:length(Ad)-1
           y(k) = y(k)-Ad(i+1)*y(k-i);
        end
        for i = 1:length(Bd)
           y(k) = y(k)+Bd(i)*u(k-i+1);
        end
    end
    
    ys = zeros(100,3);
    
    y = y';
    ruido = (0 + (0.05).*rand(N,1));
    
    ys(:,1) = y;
    
    y_ruidoso_sensor = y + ruido;
    ys(:,2) = y_ruidoso_sensor;
    y_ruidoso_dinamico = zeros(size(y));
    y_ruidoso_dinamico(1) = 0 + ruido(1);
    for k=length(Ad):100
        for i = 1:length(Ad)-1
           y_ruidoso_dinamico(k) = y_ruidoso_dinamico(k)-Ad(i+1)*y_ruidoso_dinamico(k-i);
        end
        for i = 1:length(Bd)
           y_ruidoso_dinamico(k) = y_ruidoso_dinamico(k)+Bd(i)*u(k-i+1);
        end
        y_ruidoso_dinamico(k) = y_ruidoso_dinamico(k) + ruido(k);
    end
    ys(:,3) = y_ruidoso_dinamico;
    
    tipo_saida = {'SemRuido', 'RuidoSensor', 'RuidoDinamico'};
    
    for tsaida = 1:3
        
        disp(tipo_saida(tsaida))
        for n = 1:5

            P = [];
            for j = 0:n-1
                P = [P ys(n-j:N-(j+1),tsaida)];
            end
            for j = 0:n-1
                P = [P u(n-j:N-(j+1))];
            end

            T = inv(P'*P)*P'*ys(n+1:end,tsaida);

            MSE_results(strcat(num2str(idx),num2str(n))) = T;

            y_est = P * T;

            y_est = [ys(1:n,tsaida); y_est];
            y_mean = mean(y_est);

            erro = ys(:,tsaida) - y_est;

            MSE = sum(erro.^2);
            COEFF = 1 - sum(erro.^2)/sum((ys(:,tsaida) - y_mean).^2);
            disp(strcat(num2str(n), 'ordem'))

            disp(MSE)
            disp(COEFF)
        end
     end
        % Fim da questão 2
             
     
        
     figure
     y = y + (3 + (1)*rand(1,1));
     for n = 1:5
         
        e = (0 + (0.05).*rand(N,1));
        COEFFs = [];
        MSEs = [];
        for it = 1:50

            P = [];
            for j = 0:n-1
                P = [P y(n-j:N-(j+1))];
            end
            for j = 0:n-1
                P = [P u(n-j:N-(j+1))];
            end
            for j = 0:n-1
                P = [P e(n-j:N-(j+1))];
            end

            T = inv(P'*P)*P'*y(n+1:end);

            MSE_results(strcat(num2str(idx),num2str(n))) = T;

            y_est = P * T;

            y_est = [y(1:n); y_est];
            y_mean = mean(y_est);

            erro = y - y_est;
            e = erro;

            MSE = sum(erro.^2);
            COEFF = 1 - sum(erro.^2)/sum((y - y_mean).^2);
            
            %disp(MSE)
            %disp(COEFF)
            COEFFs = [COEFFs COEFF];
            MSEs = [MSEs MSE];
        end
        disp(strcat(num2str(n), 'ordem'))
        e
        
        subplot(3,2,n);
        plot(COEFFs);
        title(strcat('Ordem ', num2str(n)));
        
     end  
end
    
     %% QUESTAO 4
 for i = 1:2
     disp(strcat('Dados', num2str(i)))
    filename = ['dados_' num2str(i) '.txt'];
    data{i} = load(filename);

     data_cell = cell2mat(data(1,1));
     u_d = data_cell(:,2);
     y_d = data_cell(:,1);

     COEFFs = [];
     MSEs = [];

     % QUESTAO 4 ARX
     for n = 1:5

        P = [];
        for j = 0:n-1
            P = [P y_d(n-j:N-(j+1))];
        end
        for j = 0:n-1
            P = [P u_d(n-j:N-(j+1))];
        end

        T = inv(P'*P)*P'*y_d(n+1:end);

        MSE_results(strcat(num2str(idx),num2str(n))) = T;

        y_est = P * T;

        y_est = [y_d(1:n); y_est];
        y_mean = mean(y_est);

        erro = y_d - y_est;

        MSE = sum(erro.^2);
        COEFF = 1 - sum(erro.^2)/sum((y_d - y_mean).^2);
        %disp(strcat(num2str(n), 'ordem'))

        COEFFs = [COEFFs COEFF];
        MSEs = [MSEs MSE];
     end
    disp('ARX');
    disp(MSEs);
    disp(COEFFs);

     % QUESTAO 4 ARMAX

     COEFFs = [];
     MSEs = [];
     for n = 1:5
        N = 500;
        e = (0 + (0.05).*rand(N,1));
        
        for it = 1:50

            P = [];
            for j = 0:n-1
                P = [P y_d(n-j:N-(j+1))];
            end

            for j = 0:n-1
                P = [P u_d(n-j:N-(j+1))];
            end

            for j = 0:n-1
                P = [P e(n-j:N-(j+1))];
            end

            T = inv(P'*P)*P'*y_d(n+1:end);

            MSE_results(strcat(num2str(idx),num2str(n))) = T;

            y_est = P * T;

            y_est = [y_d(1:n); y_est];
            y_mean = mean(y_est);

            erro = y_d - y_est;
            e = erro;

            MSE = sum(erro.^2);
            COEFF = 1 - sum(erro.^2)/sum((y_d - y_mean).^2);

            %disp(MSE)
            %disp(COEFF)
            
        end
        COEFFs = [COEFFs COEFF];
        MSEs = [MSEs MSE];
        %disp(strcat(num2str(n), 'ordem'))


        %subplot(3,2,n);
        %plot(MSEs);
        %title(strcat('Ordem ', num2str(n)));

     end   
     disp('ARMAX');
    disp(MSEs);
    disp(COEFFs)
 end







