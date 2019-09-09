function phi = montaMatReg(ordem, y, u, isArx, e)

    N = length(y);
    phi = [];
    
    if isArx
        for j = 0:ordem-1
            phi = [phi y(ordem-j:N-(j+1))];
        end

        for j = 0:ordem-1
            phi = [phi u(ordem-j:N-(j+1))];
        end
    else      
        for j = 0:ordem-1
            phi = [phi y(ordem-j:N-(j+1))];
        end

        for j = 0:ordem-1
            phi = [phi u(ordem-j:N-(j+1))];
        end

        for j = 0:ordem-1
            phi = [phi e(ordem-j:N-(j+1))];
        end
    end
end