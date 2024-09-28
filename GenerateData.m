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

% numerical and stimulation parameters
dt = 0.25;
endtime = 2000;
nsteps = ceil(endtime/dt);
stimdur= 2;
nstimdur = ceil(stimdur/dt);
spiraltime=250;
nspiraltime=ceil(spiraltime/dt);
dx=0.05;
diff=0.001; % diffusion coefficient
nx=250;
ny=nx;
dt_o_dx2=dt/(dx*dx);
%paceevery=270;

% initial values for state variables
v = 0*ones(nx,ny); %0
h = 0.5*ones(nx,ny);

% arrays for saving data
vsave=zeros(nsteps,nx,ny);
hsave=zeros(nsteps,nx,ny);
t = dt:dt:endtime;
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
    
    vsave(ntime,:,:) = v;
    hsave(ntime,:,:) = h;
    
    if(mod(ntime,200)==1)
        pcolor(v),shading interp,daspect([1 1 1]),caxis([0 1]),colorbar,title(["time = " num2str(ntime*dt)]),drawnow
    end
end

%colormap gray
%subplot(2,1,1)
%pcolor(xx,xx,squeeze(vsave(2,:,:))),shading interp,colorbar
%subplot(2,1,2)
%pcolor(t,xx,hsave'),shading interp,xlabel('Time'),ylabel('Space'), colorbar


