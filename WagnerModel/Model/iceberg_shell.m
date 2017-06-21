% set directories ---------------------------------------------------------
modelfull = 'ECCO_20th';
modelshort= 'E2';
%root   = '~/Dropbox/Icebergs/Iceberg_Model_to_share';  % root directory for project
%condloc= strcat(root,'/Conditions/',modelfull,'/'); % input directory
%outloc = strcat(root,'/Output/',modelfull,'/'); % output directory
root   = '/home/evankielley/IceDEF/WagnerModel';  % root directory for project
condloc= strcat(root,'/conditions/',modelfull,'/'); % input directory
outloc = strcat(root,'/output/',modelfull,'/'); % output directory
% -------------------------------------------------------------------------
% load input fields -------------------------------------------------------
tic
load(strcat(condloc,'mask'));  %landmask
load(strcat(condloc,sprintf('%s_vels_1992.mat',modelshort)));
load(strcat(condloc,sprintf('%s_sst_1992.mat',modelshort)));
load(strcat(outloc,'fixed.mat'),'ts_all','randoX_all','randoY_all');
fprintf('model data loaded \n')
toc
% -------------------------------------------------------------------------
% read in all parameters and analytic expressions for alpha and beta ------
analytic_parameters
% -------------------------------------------------------------------------
% specify the space domain ------------------------------------------------
LAT = double(vel.latw); LON = double(vel.lonw);
minLAT = min(LAT(:)); maxLAT = max(LAT(:));
minLON = min(LON(:)); maxLON = max(LON(:));
% -------------------------------------------------------------------------
% set run parameters ------------------------------------------------------
trajnum = 25;            % total number of iceberg trajectories to compute
final_t = 122;           % number of input field time steps
startrange = final_t/2;  % input field start range
tres = 3;                % time resoln such that "model Dt"="input DT"/tres
DT = 3;                  % Input fields time step
Dt = DT/tres;            % model timestep in days
dt = Dt*24*3600;         % model timestep in seconds
R = 6378*1e3;            % earth radius in m
dtR = dt/R*180/pi;       % need this ratio for distances in "drifting.m"
% -------------------------------------------------------------------------
t = 1:final_t;                  %how long is the run
nt= length(t)*tres;             %number of model timesteps
tt = linspace(1,length(t),nt);  %model time
% -------------------------------------------------------------------------
% Load Seeding fields -----------------------------------------------------
load Laurent_Seed
seed_X = repmat(Seed_X(:),[100,1]); %cycle through each location 100x
seed_Y = repmat(Seed_Y(:),[100,1]); %i.e. this can run 3600 icebergs
% -------------------------------------------------------------------------
% these are the circulation fields-----------------------------------------
uwF = vel.uw(:,:,t); vwF = vel.vw(:,:,t);   %water vels input
uaF = vel.ua(:,:,t); vaF = vel.va(:,:,t);   %air vels input
sst = sst(:,:,t);                       %sst vels input

%uaGRID = griddedInterpolant(uaF); vaGRID = griddedInterpolant(vaF);
%uwGRID = griddedInterpolant(uwF); vwGRID = griddedInterpolant(vwF);
%sstGRID = griddedInterpolant(sst);
% -------------------------------------------------------------------------
% loop over individual initial iceberg size classes -----------------------
% (classification from Bigg et al 1997) -----------------------------------
% -------------------------------------------------------------------------
bvec = 1:10;   %vector of which size classes to compute - has to be [1,10]
load bergdims

XIL = nan(length(bvec),trajnum,nt); YIL = XIL;   %set output lat/lon arrays
mXI = XIL; mYI = XIL;
VOL = XIL; DVOL = VOL;                           %set output vol/dvol arrays
mL = XIL; mW = XIL; mH = XIL;
UI = XIL; UA = XIL; UW = XIL;                    %set output zonal vel arrays
VI = XIL; VA = XIL; VW = XIL;                    %set output merid vel arrays
TE = XIL;                                        %set output SST array
Memat = XIL; Mvmat = XIL; Mbmat = XIL;           %set output melt rate arrays 

z=0;
% -------------------------------------------------------------------------
for bb = bvec
    % ---------------------------------------------------------------------
    bergsize = bb;   % current berg size class
    fprintf('run bergsize B%d \n',bergsize)
    % ---------------------------------------------------------------------

    % ---------------------------------------------------------------------
    
    % initialize the iceberg-----------------------------------------------
    L = bergdims(bergsize,1);
    W = bergdims(bergsize,2);
    H = bergdims(bergsize,3);
    % ---------------------------------------------------------------------
    
    % run drift and melt---------------------------------------------------
    tic
    mm = 0; ss = 0; ob = 0;
    for j = 1:trajnum
        if mod(j,10)==0; toc; fprintf('%d trajectories computed \n',j); end
        
        % pick a random trajectory start time (of Input field)
        % (note: this is not the best setup for validation runs, where you
        % may want specific trajectories to compare!)
        %ts = randi([0,round(startrange)],1);
        ts = ts_all(bb,j);
        tts= ts*tres;                  %trajectory start time (of model)
        lt = nt-tts;                   %trajectory run length
        
        xil = nan(1,lt); yil = xil;    %initialize output vectors
        mxi = xil; myi = xil;
        v = xil; dv = xil;
        ml = xil; mw = xil; mh = xil;
        uiv = v; uav = v; uwv = v;
        viv = v; vav = v; vwv = v;
        temp = v;
        Mev = v; Mvv = v; Mbv = v;
        
        %pick random grid seeding location (same note as above applies)
        %randoX = randi([1,length(seed_X)],1);
        randoX = randoX_all(bb,j);
        %randoY = randi([1,length(seed_Y)],1);
        randoY = randoY_all(bb,j);
        yig = seed_Y(randoY); xig = seed_X(randoX);
                        
        xil(1) = LON(xig); yil(1) = LAT(yig);   %initial lon and lat
        l = L*ones(1,lt); w = l*W/L; h = l*H/L; %initial berg dimensions
        v(1) = L*W*H; dv(1) = 0;                %initial volume and dvol
        ml(1) = L; mw(1) = W; mh(1) = H;
        
        i = 0; outofbound = 0; melted = 0;
        % now integrate as long as the iceberg is in the domain and not
        % melted and over the time period specified above
        while outofbound == 0 && melted == 0 && i<lt-1
            i = i+1;
            % this is only required if you change params seasonally -------
            day_yr = ts+i;
            % -------------------------------------------------------------
            drifting
            % -------------------------------------------------------------
            melting
            z=z+1;
        end
        ind = 1:i+1;
        
        XIL(bb,j,ind)=xil(ind); YIL(bb,j,ind)=yil(ind);   %store trajectory
        mXI(bb,j,ind) = mxi(ind); mYI(bb,j,ind) = myi(ind);
        VOL(bb,j,ind)=v(ind); DVOL(bb,j,ind)=dv(ind);     %store volume
        mL(bb,j,ind) = ml(ind); mW(bb,j,ind) = mw(ind); mH(bb,j,ind) = mh(ind);
        UI(bb,j,ind) = uiv(ind); VI(bb,j,ind) = viv(ind); %store ice vels
        UA(bb,j,ind) = uav(ind); VA(bb,j,ind) = vav(ind); %store air vels
        UW(bb,j,ind) = uwv(ind); VW(bb,j,ind) = vwv(ind); %store wat vels
        TE(bb,j,ind) =temp(ind);                       %store sst vels
        
        Memat(bb,j,ind) = Mev;                         %store melt rates
        Mvmat(bb,j,ind) = Mvv;
        Mbmat(bb,j,ind) = Mbv;
        
    end
    % ---------------------------------------------------------------------
    fprintf('%d icebergs died, %d lived, %d left the domain \n',mm,ss,ob)
    % ---------------------------------------------------------------------
    
end

mLAT = LAT; mLON = LON; mmsk = msk;

save(strcat(outloc,'output_full'),...
    'XIL','YIL','mXI','mYI','VOL','DVOL','UI','VI','UA','VA','UW','VW',...
    'mL','mW','mH','mLAT','mLON','mmsk','TE','Memat','Mvmat','Mbmat');

save(strcat(outloc,'fixed.mat'),'ts_all','randoX_all','randoY_all');
