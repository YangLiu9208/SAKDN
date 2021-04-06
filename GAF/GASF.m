function [GASF_image] = GASF(signal)
%Compute the Gramian Angular Field of a signal
%signal is the original signal, GASF_image is the Gramian Summation Angular Field of the signal
% Min-Max scaling
min_signal=min(signal);
max_signal=max(signal);
scaled_signal=(2*signal-max_signal-min_signal)/(max_signal-min_signal);
% Floating point inaccuracy!
for i=1:length(scaled_signal)
    if(scaled_signal(i)>=1)
        scaled_signal(i)=1;
    else
        scaled_signal(i)=scaled_signal(i);
    end
    if(scaled_signal(i)<=-1)
        scaled_signal(i)=-1;
    else
        scaled_signal(i)=scaled_signal(i);
    end 
end
% Polar encoding
    phi_signal=acos(scaled_signal);
    r=linspace(0,1,length(scaled_signal));
% GASF Computation (every term of the matrix)
  GASF_image=zeros(length(scaled_signal),length(scaled_signal));
  for m=1:length(scaled_signal)
      for n=1:length(scaled_signal)
          GASF_image(m,n)=cos(phi_signal(m)+phi_signal(n));
      end
  end
    
end

