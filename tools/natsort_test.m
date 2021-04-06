function natsort_test()
% Test function for NATSORT.
%
% (c) 2012-2019 Stephen Cobeldick
%
% See also NATSORT TESTFUN NATSORTFILES_TEST NATSORTROWS_TEST
fun = @natsort;
chk = testfun(fun);
%
%% Examples HTML
%
A =         {'a2', 'a10', 'a1'};
chk(A, fun, {'a1', 'a2', 'a10'}, [3,1,2])
B =         {'ver9.10', 'ver9.5', 'ver9.2', 'ver9.10.20', 'ver9.10.8'};
chk(B, fun, {'ver9.2', 'ver9.5', 'ver9.10', 'ver9.10.8', 'ver9.10.20'}, [3,2,1,5,4])
chk(A, fun, [], [3,1,2]);
chk(B, fun, [], [], {'ver',9,'.',10,'',[];'ver',9,'.',5,'',[];'ver',9,'.',2,'',[];'ver',9,'.',10,'.',20;'ver',9,'.',10,'.',8})
%
C = {'1.3','1.10','1.2'};
chk(C, fun, {'1.2', '1.3', '1.10'}, [3,1,2])
chk(C, '\d\.?\d*', fun, {'1.10', '1.2', '1.3'}, [2,3,1])
D = {'a2', 'A20', 'A1', 'a', 'A', 'a10','A2', 'a1'};
chk(D, [], 'ignorecase', fun, {'a', 'A', 'A1', 'a1', 'a2', 'A2', 'a10', 'A20'}, [4,5,3,8,1,7,6,2])
chk(D, [],  'matchcase', fun, {'A', 'A1', 'A2', 'A20', 'a', 'a1', 'a2', 'a10'}, [5,3,7,2,4,8,1,6])
E = {'2', 'a', '', '10', 'B', '1'};
chk(E, [],   'ascend', fun, {'', '1', '2', '10', 'a', 'B'}, [3,6,1,4,2,5])
chk(E, [],  'descend', fun, {'B', 'a', '10', '2', '1', ''}, [5,2,4,1,6,3])
chk(E, [], 'num<char', fun, {'', '1', '2', '10', 'a', 'B'}, [3,6,1,4,2,5])
chk(E, [], 'char<num', fun, {'', 'a', 'B', '1', '2', '10'}, [3,2,5,6,1,4])
F = {'10', '1', 'NaN', '2'};
chk(F, '(NaN|\d+)', 'num<NaN', fun, {'1', '2', '10', 'NaN'}, [2,4,1,3])
chk(F, '(NaN|\d+)', 'NaN<num', fun, {'NaN', '1', '2', '10'}, [3,2,4,1])
%
chk({'18446744073709551614', '18446744073709551615', '18446744073709551613'}, [], '%lu', fun,...
    {'18446744073709551613', '18446744073709551614', '18446744073709551615'}, [3,1,2])
%
chk({'x+NaN', 'x11.5', 'x-1.4', 'x', 'x-Inf', 'x+0.3'}, '[-+]?(NaN|Inf|\d+\.?\d*)', fun,... G
    {'x', 'x-Inf', 'x-1.4', 'x+0.3', 'x11.5', 'x+NaN'}, [4,5,3,6,2,1])
chk({'0.56e007', '', '43E-2', '10000', '9.8'}, '\d+\.?\d*([eE][-+]?\d+)?', fun,... H
    {'', '43E-2', '9.8', '10000', '0.56e007'}, [2,3,5,4,1])
chk({'a0X7C4z', 'a0X5z', 'a0X18z', 'a0XFz'}, '0X[0-9A-F]+', '%x', fun,... I
    {'a0X5z', 'a0XFz', 'a0X18z', 'a0X7C4z'}, [2,4,3,1])
chk({'a11111000100z', 'a101z', 'a000000000011000z', 'a1111z'}, '[01]+', '%b', fun,... J
    {'a101z', 'a1111z', 'a000000000011000z', 'a11111000100z'}, [2,4,3,1])
%
%% Examples Mfile Help
%
X =  {'x2','x10','x1'};
chk(X, fun, {'x1','x2','x10'})
chk(X, fun, [], [], {'x',2;'x',10;'x',1})
chk(X, fun, {'x1','x2','x10'}, [3,1,2], {'x',2;'x',10;'x',1}) % not in help
%
chk({'v10.6', 'v9.10', 'v9.5', 'v10.10', 'v9.10.20', 'v9.10.8'}, fun,... A
    {'v9.5', 'v9.10', 'v9.10.8', 'v9.10.20', 'v10.6', 'v10.10'}, [3,2,6,5,1,4])
chk({'test+NaN', 'test11.5', 'test-1.4', 'test', 'test-Inf', 'test+0.3'}, '[-+]?(Inf|\d+\.?\d*)', fun,... B
    {'test', 'test-Inf', 'test-1.4', 'test+0.3', 'test11.5', 'test+NaN'}, [4,5,3,6,2,1])
chk({'0.56e007', '', '43E-2', '10000', '9.8'}, '\d+\.?\d*([eE][-+]?\d+)?', fun,... C
    {''  '43E-2'  '9.8'  '10000'  '0.56e007'}, [2,3,5,4,1])
chk({'a0X7C4z', 'a0X5z', 'a0X18z', 'a0XFz'}, '0X[0-9A-F]+', '%i', fun,... D
    {'a0X5z'  'a0XFz'  'a0X18z'  'a0X7C4z'}, [2,4,3,1])
chk({'a11111000100z', 'a101z', 'a000000000011000z', 'a1111z'}, '[01]+', '%b', fun,... E
    {'a101z'  'a1111z'  'a000000000011000z'  'a11111000100z'}, [2,4,3,1])
F = {'a2', 'A20', 'A1', 'a10', 'A2', 'a1'};
chk(F, [], 'ignorecase', fun, {'A1'  'a1'  'a2'  'A2'  'a10'  'A20'}, [3,6,1,5,4,2]) % default
chk(F, [], 'matchcase',  fun, {'A1'  'A2'  'A20'  'a1'  'a2'  'a10'}, [3,5,2,6,1,4])
G =  {'2', 'a', '', '3', 'B', '1'};
chk(G, [], 'ascend', fun, {'','1','2','3','a','B'}, [3,6,1,4,2,5]) % default
chk(G, [],'descend', fun, {'B','a','3','2','1',''}, [5,2,4,1,6,3])
chk(G, [], 'num<char', fun, {'','1','2','3','a','B'}, [3,6,1,4,2,5]) % default
chk(G, [], 'char<num', fun, {'','a','B','1','2','3'}, [3,2,5,6,1,4])
%
chk({'a18446744073709551615z', 'a18446744073709551614z'}, '%lu', fun,...
    {'a18446744073709551614z', 'a18446744073709551615z'}, [2,1])
%
%% Examples Number Substring Table
%
idf = @(varargin) [repmat({''},numel(varargin),1),varargin(:)];
%
% unsigned integer:
chk({'0','123','4','56789'}, '\d+', '%f', fun, {'0','4','123','56789'}, [1,3,2,4], idf(0,123,4,56789))
chk({'0','123','4','56789'}, '\d+', '%i', fun, {'0','4','123','56789'}, [1,3,2,4], idf(0,123,4,56789))
chk({'0','123','4','56789'}, '\d+', '%u', fun, {'0','4','123','56789'}, [1,3,2,4], idf(0,123,4,56789))
chk({'0','123','4','56789'}, '\d+', '%lu', fun, {'0','4','123','56789'}, [1,3,2,4], idf(0,123,4,56789))
% signed integer:
chk({'+1','23','-45','678'}, '[-+]?\d+', '%f', fun, {'-45','+1','23','678'}, [3,1,2,4], idf(1,23,-45,678))
chk({'+1','23','-45','678'}, '[-+]?\d+', '%i', fun, {'-45','+1','23','678'}, [3,1,2,4], idf(1,23,-45,678))
chk({'+1','23','-45','678'}, '[-+]?\d+', '%d', fun, {'-45','+1','23','678'}, [3,1,2,4], idf(1,23,-45,678))
chk({'+1','23','-45','678'}, '[-+]?\d+', '%ld', fun, {'-45','+1','23','678'}, [3,1,2,4], idf(1,23,-45,678))
% floating point:
chk({'012','3.45','678.9'}, '\d+\.?\d*', '%f', fun, {'3.45','012','678.9'}, [2,1,3], idf(12,3.45,678.9))
chk({'123','4','NaN','Inf'}, '(\d+|Inf|NaN)', '%f', fun, {'4','123','Inf','NaN'}, [2,1,4,3], idf(123,4,NaN,Inf))
chk({'0.123e4','5.67e08'}, '\d+\.\d+e\d+', fun, {'0.123e4','5.67e08'}, [1,2], idf(0.123e4,5.67e08))
% octal:
chk({'012','03456','0700'}, '0[0-7]+', '%i', fun, {'012','0700','03456'}, [1,3,2], idf(10,1838,448))
chk({'012','03456','0700'}, '0[0-7]+', '%o', fun, {'012','0700','03456'}, [1,3,2], idf(10,1838,448))
chk({ '12', '3456', '700'},  '[0-7]+', '%o', fun, { '12', '700', '3456'}, [1,3,2], idf(10,1838,448))
% hexadecimal:
chk({'0X0','0X3E7','0XFF'}, '0X[0-9A-F]+', '%i', fun, {'0X0','0XFF','0X3E7'}, [1,3,2], idf(0,999,255))
chk({'0X0','0X3E7','0XFF'}, '0X[0-9A-F]+', '%x', fun, {'0X0','0XFF','0X3E7'}, [1,3,2], idf(0,999,255))
chk({  '0',  '3E7',  'FF'},   '[0-9A-F]+', '%x', fun, {  '0',  'FF',  '3E7'}, [1,3,2], idf(0,999,255))
% binary:
chk({'0B1','0B101','0B10'}, '0B[01]+', '%b', fun, {'0B1','0B10','0B101'}, [1,3,2], idf(1,5,2))
chk({  '1',  '101',  '10'},   '[01]+', '%b', fun, {  '1',  '10',  '101'}, [1,3,2], idf(1,5,2))
%
%% Numbers and NaN
%
H = {'aa','a1','10','1','','ac','2a','a10','ab','2','a2','a','10a','1a','c','b'};
chk(H, [], 'num<char','ascend', fun,...
    {'','1','1a','2','2a','10','10a','a','a1','a2','a10','aa','ab','ac','b','c'}, [5,4,14,10,7,3,13,12,2,11,8,1,9,6,16,15])
chk(H, [], 'num<char', 'descend', fun,...
    {'c','b','ac','ab','aa','a10','a2','a1','a','10a','10','2a','2','1a','1',''}, [15,16,6,9,1,8,11,2,12,13,3,7,10,14,4,5])
chk(H, [], 'char<num', 'ascend', fun,...
    {'','a','aa','ab','ac','a1','a2','a10','b','c','1','1a','2','2a','10','10a'}, [5,12,1,9,6,2,11,8,16,15,4,14,10,7,3,13])
chk(H, [], 'char<num', 'descend', fun,...
    {'10a','10','2a','2','1a','1','c','b','a10','a2','a1','ac','ab','aa','a',''}, [13,3,7,10,14,4,15,16,8,11,2,6,9,1,12,5])
%
I = {'a','1','b','10','','2','a2','a10','a1','a1y','a1z','a1x9','a1x10','a1x1','a1x'};
chk(I, [], 'num<char', 'ascend', fun,...
    {'','1','2','10','a','a1','a1x','a1x1','a1x9','a1x10','a1y','a1z','a2','a10','b'}, [5,2,6,4,1,9,15,14,12,13,10,11,7,8,3])
chk(I, [], 'num<char', 'descend', fun,...
    {'b','a10','a2','a1z','a1y','a1x10','a1x9','a1x1','a1x','a1','a','10','2','1',''}, [3,8,7,11,10,13,12,14,15,9,1,4,6,2,5])
chk(I, [], 'char<num', 'ascend', fun,...
    {'','a','a1','a1x','a1x1','a1x9','a1x10','a1y','a1z','a2','a10','b','1','2','10'}, [5,1,9,15,14,12,13,10,11,7,8,3,2,6,4])
chk(I, [], 'char<num', 'descend', fun,...
    {'10','2','1','b','a10','a2','a1z','a1y','a1x10','a1x9','a1x1','a1x','a1','a',''}, [4,6,2,3,8,7,11,10,13,12,14,15,9,1,5])
%
J = {'aaa','111','a11','1a1','aa1','11a','a1a','1aa'};
chk(J, [], 'num<char', 'ascend', fun,...
    {'1a1','1aa','11a','111','a1a','a11','aa1','aaa'}, [4,8,6,2,7,3,5,1])
chk(J, [], 'num<char', 'descend', fun,...
    {'aaa','aa1','a11','a1a','111','11a','1aa','1a1'}, [1,5,3,7,2,6,8,4])
chk(J, [], 'char<num', 'ascend', fun,...
    {'aaa','aa1','a1a','a11','1aa','1a1','11a','111'}, [1,5,7,3,8,4,6,2])
chk(J, [], 'char<num', 'descend', fun,...
    {'111','11a','1a1','1aa','a11','a1a','aa1','aaa'}, [2,6,4,8,3,7,5,1])
%
K = {'1234','1200','129'};
chk(K, '\d{1,2}', fun,... quantifier
    {'1200','129','1234'}, [2,3,1], {'',12,'',34;'',12,'',0;'',12,'',9})
chk(K, '(?<=\d{2})\d+', fun,... lookaround assertion
    {'1200','129','1234'}, [2,3,1], {'12',34;'12',0;'12',9})
%
M = {'10','NaNb','NaN','NaNc','1','NaNNaN','2','NaNa','NaN','10'};
N = {'',10,'','';'',NaN,'b','';'',NaN,'','';'',NaN,'c','';'',1,'','';'',NaN,'',NaN;'',2,'','';'',NaN,'a','';'',NaN,'','';'',10,'',''};
chk(M, '(NaN|\d+)', 'num<NaN', 'ascend', fun,...
    {'1','2','10','10','NaN','NaN','NaNNaN','NaNa','NaNb','NaNc'}, [5,7,1,10,3,9,6,8,2,4], N)
chk(M, '(NaN|\d+)', 'num<NaN', 'descend', fun,...
    {'NaNc','NaNb','NaNa','NaNNaN','NaN','NaN','10','10','2','1'}, [4,2,8,6,3,9,1,10,7,5], N)
chk(M, '(NaN|\d+)', 'NaN<num', 'ascend', fun,...
    {'NaN','NaN','NaNNaN','NaNa','NaNb','NaNc','1','2','10','10'}, [3,9,6,8,2,4,5,7,1,10], N)
chk(M, '(NaN|\d+)', 'NaN<num', 'descend', fun,...
    {'10','10','2','1','NaNc','NaNb','NaNa','NaNNaN','NaN','NaN'}, [1,10,7,5,4,2,8,6,3,9], N)
%
%% Orientation
%
chk({}, fun, {}, [], {}) % empty!
chk(cell(0,2,3), fun, cell(0,2,3), nan(0,2,3)) % empty!
chk({'1';'10';'20';'2'}, fun,...
    {'1';'2';'10';'20'}, [1;4;2;3])
chk({'2','10','8';'#','a',' '}, fun,...
    {'2','10','#';'8',' ','a'}, [1,3,2;5,6,4])
%
%% Stability
%
chk({'';'';''}, fun, {'';'';''}, [1;2;3], {'';'';''})
%
U = {'2';'3';'2';'1';'2'};
chk(U, [], 'ascend', fun,...
    {'1';'2';'2';'2';'3'}, [4;1;3;5;2])
chk(U, [], 'descend', fun,...
    {'3';'2';'2';'2';'1'}, [2;1;3;5;4])
%
V = {'x';'z';'y';'';'z';'';'x';'y'};
chk(V, [], 'ascend', fun,...
    {'';'';'x';'x';'y';'y';'z';'z'},[4;6;1;7;3;8;2;5])
chk(V, [], 'descend', fun,...
    {'z';'z';'y';'y';'x';'x';'';''},[2;5;3;8;1;7;4;6])
%
W = {'2x';'2z';'2y';'2';'2z';'2';'2x';'2y'};
chk(W, [], 'ascend', fun,...
    {'2';'2';'2x';'2x';'2y';'2y';'2z';'2z'},[4;6;1;7;3;8;2;5])
chk(W, [], 'descend', fun,...
    {'2z';'2z';'2y';'2y';'2x';'2x';'2';'2'},[2;5;3;8;1;7;4;6])
%
%% Other Implementation Examples
%
% <https://code.activestate.com/recipes/285264-natural-string-sorting/>
chk({'Team 11','Team 3','Team 1'}, fun,...
    {'Team 1','Team 3','Team 11'})
chk({'ver-1.3.12','ver-1.3.3','ver-1.2.5','ver-1.2.15','ver-1.2.3','ver-1.2.1'}, fun,...
    {'ver-1.2.1','ver-1.2.3','ver-1.2.5','ver-1.2.15','ver-1.3.3','ver-1.3.12'})
chk({'C1H2','C1H4','C2H2','C2H6','C2N','C3H6'}, fun,...
    {'C1H2','C1H4','C2H2','C2H6','C2N','C3H6'})
chk({'Team 101','Team 58','Team 30','Team 1'}, fun,...
    {'Team 1','Team 30','Team 58','Team 101'})
chk({'a5','A7','a15','a9','A8'}, fun,...
    {'a5','A7','A8','a9','a15'})
%
% <http://www.davekoelle.com/alphanum.html>
chk({'1000X Radonius Maximus','10X Radonius','200X Radonius','20X Radonius','20X Radonius Prime','30X Radonius','40X Radonius','Allegia 50 Clasteron','Allegia 500 Clasteron','Allegia 50B Clasteron','Allegia 51 Clasteron','Allegia 6R Clasteron','Alpha 100','Alpha 2','Alpha 200','Alpha 2A','Alpha 2A-8000','Alpha 2A-900','Callisto Morphamax','Callisto Morphamax 500','Callisto Morphamax 5000','Callisto Morphamax 600','Callisto Morphamax 6000 SE','Callisto Morphamax 6000 SE2','Callisto Morphamax 700','Callisto Morphamax 7000','Xiph Xlater 10000','Xiph Xlater 2000','Xiph Xlater 300','Xiph Xlater 40','Xiph Xlater 5','Xiph Xlater 50','Xiph Xlater 500','Xiph Xlater 5000','Xiph Xlater 58'}, fun,...
    {'10X Radonius','20X Radonius','20X Radonius Prime','30X Radonius','40X Radonius','200X Radonius','1000X Radonius Maximus','Allegia 6R Clasteron','Allegia 50 Clasteron','Allegia 50B Clasteron','Allegia 51 Clasteron','Allegia 500 Clasteron','Alpha 2','Alpha 2A','Alpha 2A-900','Alpha 2A-8000','Alpha 100','Alpha 200','Callisto Morphamax','Callisto Morphamax 500','Callisto Morphamax 600','Callisto Morphamax 700','Callisto Morphamax 5000','Callisto Morphamax 6000 SE','Callisto Morphamax 6000 SE2','Callisto Morphamax 7000','Xiph Xlater 5','Xiph Xlater 40','Xiph Xlater 50','Xiph Xlater 58','Xiph Xlater 300','Xiph Xlater 500','Xiph Xlater 2000','Xiph Xlater 5000','Xiph Xlater 10000'})
%
% <https://natsort.readthedocs.io/en/master/examples.html>
chk({'2 ft 7 in', '1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '7 ft 6 in'}, fun,...
    {'1 ft 5 in', '2 ft 7 in', '2 ft 11 in', '7 ft 6 in', '10 ft 2 in'})
chk({'version-1.9', 'version-2.0', 'version-1.11', 'version-1.10'}, fun,...
    {'version-1.9', 'version-1.10', 'version-1.11', 'version-2.0'})
chk({'position5.10.data', 'position-3.data', 'position5.3.data', 'position2.data'}, '[-+]?\d+\.?\d*', fun,...
    {'position-3.data', 'position2.data', 'position5.10.data', 'position5.3.data'})
chk({'1.2', '1.2rc1', '1.2beta2', '1.2beta1', '1.2alpha', '1.2.1', '1.1', '1.3'}, fun,...
    {'1.1', '1.2', '1.2.1', '1.2alpha', '1.2beta1', '1.2beta2', '1.2rc1', '1.3'})
a = {'a50', 'a51.', 'a+50.4', 'a5.034e1', 'a+50.300'};
chk(a, '\d+\.?\d*(E\d+)?',      fun, {'a50', 'a5.034e1', 'a51.', 'a+50.300', 'a+50.4'}) % no sign
chk(a, '[-+]?\d+\.?\d*(E\d+)?', fun, {'a50', 'a+50.300', 'a5.034e1', 'a+50.4', 'a51.'})
chk(a, '[-+]?\d+\.?\d*',        fun, {'a5.034e1', 'a50', 'a+50.300', 'a+50.4', 'a51.'}) % no exp
a =  {'a2', 'a9', 'a1', 'a4', 'a10'};
chk(a, fun, {'a1', 'a2', 'a4', 'a9', 'a10'}, [3,1,4,2,5])
chk(a, [], 'descend', fun, {'a10', 'a9', 'a4', 'a2', 'a1'})
%
chk() % display summary
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%natsort_test