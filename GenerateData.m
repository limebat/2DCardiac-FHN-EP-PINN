clc
clear
clf

% Mitchell-Schaeffer model in 2d
% from 2003 Bulletin of Mathematical Biology

% parameter values 
tau_in=0.3;
tau_out=6;
% original value; decreasing tau_open promotes breakup
tau_open=120;
%tau_open = 80;
%tau_close = 200;
% original value; increasing tau_close cauess breakup a little earlier
tau_close=150;
v_gate=0.13; %0.13 %0.35 gives oscillatory

v_stim=0.056; %twice diastolic threshold for 2ms duration

convolution_factor = 5;

% numerical and stimulation parameters
dt = 0.25;
endtime = 350;
nsteps = ceil(endtime/dt);
stimdur= 2;
nstimdur = ceil(stimdur/dt);
spiraltime=250;
nspiraltime=ceil(spiraltime/dt);
dx=0.05*convolution_factor; %*5^(1/2)
diff=0.001; % diffusion coefficient
nx=250/convolution_factor;
ny=nx;
dt_o_dx2=dt/(dx*dx);
%paceevery=270;

% initial values for state variables
v = 0*ones(nx,ny); %0
h = 0.5*ones(nx,ny); 

fileID = fopen('TimeVH.txt', 'w');
fileIDMESH = fopen('XY.txt', 'w');

fprintf(fileID, 'sec, V, H\n'); 
fprintf(fileIDMESH, 'X, Y\n'); 


%% Save initial state data at t = 0
%data_row_0 = [0, v(:)', h(:)']; % Flatten v and h and include time = 0
%fprintf(fileID, '%g', data_row_0(1)); % Print the first value without a leading comma
%for j = 2:length(data_row_0)
%    fprintf(fileID, ', %g', data_row_0(j)); % Print subsequent values with a leading comma
%end
%fprintf(fileID, '\n'); % Move to the next line

%Begin recording time and their iterations
record_iterations_begin = 200;
t = 0:dt:endtime;
xx=1:nx;
xx=xx*dx;

% time loop
for ntime=1:nsteps

    % apply stimulus if it's time
%    if(mod(ntime,ceil(paceevery/dt))<nstimdur)
    if(ntime==1)
		v(:,1:10) = 0.5;
    end
    if(ntime==nspiraltime)
        v(1:ceil(nx/2),:)=0;
    end
	jstim=0;
%    if(ntime<=nstimdur && )
%        jstim=v_stim;
%    else
%        jstim=0;
%    end
    
    % calculate currents
    jin=h.*v.*v.*(1-v)/tau_in;
    jout=-v/tau_out;
    
    % update derivatives for state variables
    dv=jin+jout+jstim;
%    if(v<v_gate)
%        dh=(1.-h)/tau_open;
%    else
%        dh=-h/tau_close;
%    end
	dh=(v<v_gate).*((1.-h)/tau_open)+(v>=v_gate).*(-h/tau_close);

    xlap=zeros(nx,ny);
    for j=1:ny
        for i=1:nx
            if(i==1)
                xlap1=2*(v(2,j)-v(1,j));
            elseif(i==nx)
                xlap1=2*(v(nx-1,j)-v(nx,j));
            else
                xlap1=v(i-1,j)-2*v(i,j)+v(i+1,j);
            end
            if(j==1)
                xlap2=2*(v(i,2)-v(i,1));
            elseif(j==ny)
                xlap2=2*(v(i,ny-1)-v(i,ny));
            else
                xlap2=v(i,j-1)-2*v(i,j)+v(i,j+1);
            end
%            xlap(i,j)=diff*dt_o_dx2*(xlap1+xlap2);
            xlap(i,j)=xlap1+xlap2;
%            xlap(i,j)=xlap(i,j)*diff*dt_o_dx2;
        end
    end
    xlap=xlap*diff*dt_o_dx2;

    % integrate using forward Euler method
%    v = v + dt*dv + diff*dt/dx/dx*xlap;
%    v = v + dt*dv + diff*dt_o_dx2*xlap;
    v = v + dt*dv + xlap;
    h = h + dt*dh;
    if mod(ntime*dt, 5) == 0 && ntime*dt > record_iterations_begin
        pcolor(v),shading interp,daspect([1 1 1]),caxis([0 1]),colorbar,title(["time = " num2str(ntime*dt)]),drawnow

        % Save state variables
        
        % Flatten and save to file
        v_flatten = v(:)';
        h_flatten = h(:)';
        data_row = [ntime*dt, v_flatten, h_flatten];
        % Print the data row without an extra comma at the end
        fprintf(fileID, '%g', data_row(1)); % Print the first value without a leading comma
        for j = 2:length(data_row)
            fprintf(fileID, ', %g', data_row(j)); % Print subsequent values with a leading comma
        end
        fprintf(fileID, '\n'); % Move to the next line
            
    end
end

[x, y] = meshgrid(xx, xx);

mesh = [x(:), y(:)];

fprintf(fileIDMESH, '%g, %g\n', mesh'); % Adjusted to ensure each point is on a new line
fprintf(fileIDMESH, '\n');

fclose(fileID);
fclose(fileIDMESH);

disp('Data saved to TimeVH.txt');

%colormap gray
%subplot(2,1,1)
%pcolor(xx,xx,squeeze(vsave(2,:,:))),shading interp,colorbar
%subplot(2,1,2)
%pcolor(t,xx,hsave'),shading interp,xlabel('Time'),ylabel('Space'), colorbar


