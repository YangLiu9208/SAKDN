%% NATSORT Examples
% The function <https://www.mathworks.com/matlabcentral/fileexchange/34464
% |NATSORT|> sorts a cell array of strings (1xN char), taking into account
% any number values within the strings. This is known as a _natural order
% sort_ or an _alphanumeric sort_. Note that MATLAB's inbuilt
% <https://www.mathworks.com/help/matlab/ref/sort.html |SORT|> function sorts
% only by character order, as does |SORT| in most programming languages.
%
% For sorting filenames or filepaths use
% <https://www.mathworks.com/matlabcentral/fileexchange/47434 |NATSORTFILES|>.
%
% For sorting the rows of a cell array of strings use
% <https://www.mathworks.com/matlabcentral/fileexchange/47433 |NATSORTROWS|>.
%
%% Basic Usage: Integer Numbers
% By default |NATSORT| interprets consecutive digits as being part of a
% single integer, any remaining substring/s are treated as character:
A = {'a2', 'a10', 'a1'};
sort(A)
natsort(A)
B = {'ver9.10', 'ver9.5', 'ver9.2', 'ver9.10.20', 'ver9.10.8'};
sort(B)
natsort(B)
%% Output 2: Sort Index
% The second output argument is a numeric array of the sort indices |ndx|,
% such that |Y = X(ndx)| where |Y = natsort(X)|:
[~,ndx] = natsort(A)
%% Output 3: Debugging Array
% The third output is a cell array which contains all matched numbers
% (after converting to numeric using the specified |SSCANF| format) and
% all split character substrings. This cell array is useful for
% confirming that the numbers are being correctly identified by the
% regular expression. Note that the rows of the debugging cell array are
% <https://www.mathworks.com/company/newsletters/articles/matrix-indexing-in-matlab.html
% linearly indexed> from the input cell array, the even columns contain any
% matched number values, while the odd columns contain any split substrings:
[~,~,dbg] = natsort(B)
%% Input 2: Regular Expression
% The optional second input argument is a regular expression which
% specifies the number matching:
C = {'1.3','1.10','1.2'};
natsort(C)   % by default match integers
natsort(C, '\d+\.?\d*') % match decimal fractions
%% Input 3+: Case Sensitivity
% By default |NATSORT| provides a case-insensitive sort of the input strings.
% An optional input argument selects case-sensitive/insensitive sorting:
D = {'a2', 'A20', 'A1', 'a', 'A', 'a10','A2', 'a1'};
natsort(D, [], 'ignorecase') % default
natsort(D, [], 'matchcase')
%% Input 3+: Sort Direction
% By default |NATSORT| provides an ascending sort of the input strings.
% An optional input argument selects the sort direction (note that
% characters and numbers are either both ascending or both descending):
E = {'2', 'a', '', '10', 'B', '1'};
natsort(E, [], 'ascend') % default
natsort(E, [], 'descend')
%% Input 3+: Char/Number Order
% By default |NATSORT| sorts characters after numbers.
% An optional input argument selects if characters are treated as
% _greater-than_ or _less-than_ numbers:
natsort(E, [], 'num<char') % default
natsort(E, [], 'char<num')
%% Input 3+: NaN/Number Order
% By default |NATSORT| sorts NaN after all other numbers.
% An optional input argument selects if NaN are treated as
% _greater-than_ or _less-than_ numbers:
F = {'10', '1', 'NaN', '2'};
natsort(F, '(NaN|\d+)', 'num<NaN') % default
natsort(F, '(NaN|\d+)', 'NaN<num')
%% Input 3+: |SSCANF| Format String (Floating Point, Hexadecimal, Octal, Binary, 64 Bit Integer)
% The default format string |'%f'| will correctly parse many common number
% types: this includes decimal integers, decimal fractions, |NaN|, |Inf|,
% and numbers written in E-notation. For hexadecimal, octal, binary, and
% 64-bit integers the format string must be specified as an input argument.
% Supported <https://www.mathworks.com/help/matlab/ref/sscanf.html
% |SSCANF|> formats are shown in this table:
%
% <html>
% <table>
%  <tr><th>Format String</th><th>Number Types</th></tr>
%  <tr><td>%e, %f, %g</td>   <td>floating point numbers</td></tr>
%  <tr><td>%d</td>           <td>signed decimal</td></tr>
%  <tr><td>%i</td>           <td>signed decimal, octal, or hexadecimal</td></tr>
%  <tr><td>%ld, %li</td>     <td>signed 64 bit, decimal, octal, or hexadecimal</td></tr>
%  <tr><td>%u</td>           <td>unsigned decimal</td></tr>
%  <tr><td>%o</td>           <td>unsigned octal</td></tr>
%  <tr><td>%x</td>           <td>unsigned hexadecimal</td></tr>
%  <tr><td>%lu, %lo, %lx</td><td>unsigned 64-bit decimal, octal, or hexadecimal</td></tr>
%  <tr><td>%b</td>           <td>unsigned binary integer (custom parsing, not SSCANF)</td></tr>
% </table>
% </html>
%
% For example large
% integers can be converted to 64-bit numerics, with their full precision:
natsort({'18446744073709551614', '18446744073709551615', '18446744073709551613'}, [], '%lu')
%% Regular Expression: Decimal Fractions, E-notation, +/- Sign
% |NATSORT| relies on <https://www.mathworks.com/help/matlab/ref/regexpi.html
% |REGEXPI|> to detect numbers in the strings. In order to match
% the required number format (e.g. decimal fractions, exponents,
% or a positive/negative sign, etc.) simply provide a suitable
% <https://www.mathworks.com/help/matlab/matlab_prog/regular-expressions.html
% regular expression> as an optional input argument:
G = {'x+NaN', 'x11.5', 'x-1.4', 'x', 'x-Inf', 'x+0.3'};
sort(G)
natsort(G, '[-+]?(NaN|Inf|\d+\.?\d*)')
H = {'0.56e007', '', '43E-2', '10000', '9.8'};
sort(H)
natsort(H, '\d+\.?\d*([eE][-+]?\d+)?')
%% Regular Expression: Hexadecimal, Octal, Binary Integers
% Integers encoded in hexadecimal, octal, or binary may also be parsed and
% sorted correctly. This requires both an appropriate regular expression
% that can detect the integers correctly and also a suitable |SSCANF|
% format string for converting the detected number string into numeric:
I = {'a0X7C4z', 'a0X5z', 'a0X18z', 'a0XFz'};
sort(I)
natsort(I, '0X[0-9A-F]+', '%x') % hexadecimal
J = {'a11111000100z', 'a101z', 'a000000000011000z', 'a1111z'};
sort(J)
natsort(J, '[01]+', '%b') % binary
%% Bonus: Interactive Regular Expression Tool
% Regular expressions are powerful and compact, but getting them right is
% not always easy. One assistance is to download my interactive tool
% <https://www.mathworks.com/matlabcentral/fileexchange/48930 |IREGEXP|>,
% which lets you quickly try different regular expressions and see all of
% <https://www.mathworks.com/help/matlab/ref/regexp.html |REGEXP|>'s
% outputs displayed and updated as you type.