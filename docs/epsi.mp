% ----------------------------------------------------------------------
% Evaluting Echo Planar Spectroscopic Imaging with a Columnar Excitation
% for "Virtual Biopsies": MR Pulse Sequence Graphics
% Pulse sequence graphics
%
% Author(s):
%   Michael S. Yao
%   M. Dylan Tisdall
%
% Portions of this code were adapted from Mark J White <mark@celos.net>
% and https://github.com/dangom/mr-sequence-diagrams.
% ----------------------------------------------------------------------
input mr-sequence-diagrams/pulses.mp

  filenametemplate "docs/figs-%c.mps";

% 3D gradient-echo sequence with EPI factor 3
beginfig(0);

  initf;

  startline(rf,"RF");
  xline(1);
  xpulse(1,1,"90");
  xline(2);
  xpulse(1,2, "180");
  xline(1);
  oecho(0.9);
  xacq(1.8);
  oecho(0.9);
  xacq(1.8);
  oecho(0.9);
  xacq(1.8);
  xline(0.6);

  startline(g0,"Gx (RO)");
  xline(3.8);
  xblip(0.2, -0.4);
  xline(1);
  xblip(0.2, -0.4);
  xline(0.8);
  xsquare(1.8,-0.5);
  xsquare(1.8,0.5);
  xsquare(1.8,-0.5);
  xline(0.6);

  startline(g1,"Gy (PE)");
  xline(3.8);
  xblip(0.2, -1.2);
  xsquare(1, -0.8);
  xblip(0.2, -1.2);
  xline(6.8);

  startline(g2,"Gz (SS)");
  xline(1);
  xsquare(1, -0.8);
  xsquare(0.5, 0.8);
  xline(1.3);
  xblip(0.2, -0.4);
  xline(1);
  xblip(0.2, -0.4);
  xline(6.8);

endfig;

% Echo symbol for key
beginfig(100);
  initf;
  oecho(0);
endfig;

end
