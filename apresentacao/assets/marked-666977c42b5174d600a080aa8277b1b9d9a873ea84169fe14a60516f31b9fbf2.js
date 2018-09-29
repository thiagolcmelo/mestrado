(function(){function e(e){this.tokens=[],this.tokens.links={},this.options=e||h.defaults,this.rules=o.normal,this.options.gfm&&(this.options.tables?this.rules=o.tables:this.rules=o.gfm)}function t(e,t){if(this.options=t||h.defaults,this.links=e,this.rules=a.normal,!this.links)throw new Error("Tokens array requires a `links` property.");this.options.gfm?this.options.breaks?this.rules=a.breaks:this.rules=a.gfm:this.options.pedantic&&(this.rules=a.pedantic)}function s(e){this.tokens=[],this.token=null,this.options=e||h.defaults}function n(e,t){return e.replace(t?/&/g:/&(?!#?\w+;)/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;")}function i(e,t){return e=e.source,t=t||"",function s(n,i){return n?(i=i.source||i,i=i.replace(/(^|[^\[])\^/g,"$1"),e=e.replace(n,i),s):new RegExp(e,t)}}function r(){}function l(e){for(var t,s,n=1;n<arguments.length;n++){t=arguments[n];for(s in t)Object.prototype.hasOwnProperty.call(t,s)&&(e[s]=t[s])}return e}function h(t,i,r){if(r||"function"==typeof i){r||(r=i,i=null),i&&(i=l({},h.defaults,i));var o=e.lex(o,i),a=i.highlight,u=0,p=o.length,g=0;if(!a||a.length<3)return r(null,s.parse(o,i));for(var c=function(){delete i.highlight;var e=s.parse(o,i);return i.highlight=a,r(null,e)};g<p;g++)(function(e){if("code"===e.type)return u++,a(e.text,e.lang,function(t,s){return null==s||s===e.text?--u||c():(e.text=s,e.escaped=!0,void(--u||c()))})})(o[g])}else try{return i&&(i=l({},h.defaults,i)),s.parse(e.lex(t,i),i)}catch(e){if(e.message+="\nPlease report this to https://github.com/chjj/marked.",(i||h.defaults).silent)return"<p>An error occured:</p><pre>"+n(e.message+"",!0)+"</pre>";throw e}}var o={newline:/^\n+/,code:/^( {4}[^\n]+\n*)+/,fences:r,hr:/^( *[-*_]){3,} *(?:\n+|$)/,heading:/^ *(#{1,6}) *([^\n]+?) *#* *(?:\n+|$)/,nptable:r,lheading:/^([^\n]+)\n *(=|-){3,} *\n*/,blockquote:/^( *>[^\n]+(\n[^\n]+)*\n*)+/,list:/^( *)(bull) [\s\S]+?(?:hr|\n{2,}(?! )(?!\1bull )\n*|\s*$)/,html:/^ *(?:comment|closed|closing) *(?:\n{2,}|\s*$)/,def:/^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +["(]([^\n]+)[")])? *(?:\n+|$)/,table:r,paragraph:/^((?:[^\n]+\n?(?!hr|heading|lheading|blockquote|tag|def))+)\n*/,text:/^[^\n]+/};o.bullet=/(?:[*+-]|\d+\.)/,o.item=/^( *)(bull) [^\n]*(?:\n(?!\1bull )[^\n]*)*/,o.item=i(o.item,"gm")(/bull/g,o.bullet)(),o.list=i(o.list)(/bull/g,o.bullet)("hr",/\n+(?=(?: *[-*_]){3,} *(?:\n+|$))/)(),o._tag="(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:/|@)\\b",o.html=i(o.html)("comment",/\x3c!--[\s\S]*?--\x3e/)("closed",/<(tag)[\s\S]+?<\/\1>/)("closing",/<tag(?:"[^"]*"|'[^']*'|[^'">])*?>/)(/tag/g,o._tag)(),o.paragraph=i(o.paragraph)("hr",o.hr)("heading",o.heading)("lheading",o.lheading)("blockquote",o.blockquote)("tag","<"+o._tag)("def",o.def)(),o.normal=l({},o),o.gfm=l({},o.normal,{fences:/^ *(`{3,}|~{3,}) *(\S+)? *\n([\s\S]+?)\s*\1 *(?:\n+|$)/,paragraph:/^/}),o.gfm.paragraph=i(o.paragraph)("(?!","(?!"+o.gfm.fences.source.replace("\\1","\\2")+"|")(),o.tables=l({},o.gfm,{nptable:/^ *(\S.*\|.*)\n *([-:]+ *\|[-| :]*)\n((?:.*\|.*(?:\n|$))*)\n*/,table:/^ *\|(.+)\n *\|( *[-:]+[-| :]*)\n((?: *\|.*(?:\n|$))*)\n*/}),e.rules=o,e.lex=function(t,s){var n=new e(s);return n.lex(t)},e.prototype.lex=function(e){return e=e.replace(/\r\n|\r/g,"\n").replace(/\t/g,"    ").replace(/\u00a0/g," ").replace(/\u2424/g,"\n"),this.token(e,!0)},e.prototype.token=function(e,t){for(var s,n,i,r,l,h,a,u,p,e=e.replace(/^ +$/gm,"");e;)if((i=this.rules.newline.exec(e))&&(e=e.substring(i[0].length),i[0].length>1&&this.tokens.push({type:"space"})),i=this.rules.code.exec(e))e=e.substring(i[0].length),i=i[0].replace(/^ {4}/gm,""),this.tokens.push({type:"code",text:this.options.pedantic?i:i.replace(/\n+$/,"")});else if(i=this.rules.fences.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"code",lang:i[2],text:i[3]});else if(i=this.rules.heading.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"heading",depth:i[1].length,text:i[2]});else if(t&&(i=this.rules.nptable.exec(e))){for(e=e.substring(i[0].length),h={type:"table",header:i[1].replace(/^ *| *\| *$/g,"").split(/ *\| */),align:i[2].replace(/^ *|\| *$/g,"").split(/ *\| */),cells:i[3].replace(/\n$/,"").split("\n")},u=0;u<h.align.length;u++)/^ *-+: *$/.test(h.align[u])?h.align[u]="right":/^ *:-+: *$/.test(h.align[u])?h.align[u]="center":/^ *:-+ *$/.test(h.align[u])?h.align[u]="left":h.align[u]=null;for(u=0;u<h.cells.length;u++)h.cells[u]=h.cells[u].split(/ *\| */);this.tokens.push(h)}else if(i=this.rules.lheading.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"heading",depth:"="===i[2]?1:2,text:i[1]});else if(i=this.rules.hr.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"hr"});else if(i=this.rules.blockquote.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"blockquote_start"}),i=i[0].replace(/^ *> ?/gm,""),this.token(i,t),this.tokens.push({type:"blockquote_end"});else if(i=this.rules.list.exec(e)){for(e=e.substring(i[0].length),r=i[2],this.tokens.push({type:"list_start",ordered:r.length>1}),i=i[0].match(this.rules.item),s=!1,p=i.length,u=0;u<p;u++)h=i[u],a=h.length,h=h.replace(/^ *([*+-]|\d+\.) +/,""),~h.indexOf("\n ")&&(a-=h.length,h=this.options.pedantic?h.replace(/^ {1,4}/gm,""):h.replace(new RegExp("^ {1,"+a+"}","gm"),"")),this.options.smartLists&&u!==p-1&&(l=o.bullet.exec(i[u+1])[0],r===l||r.length>1&&l.length>1||(e=i.slice(u+1).join("\n")+e,u=p-1)),n=s||/\n\n(?!\s*$)/.test(h),u!==p-1&&(s="\n"===h[h.length-1],n||(n=s)),this.tokens.push({type:n?"loose_item_start":"list_item_start"}),this.token(h,!1),this.tokens.push({type:"list_item_end"});this.tokens.push({type:"list_end"})}else if(i=this.rules.html.exec(e))e=e.substring(i[0].length),this.tokens.push({type:this.options.sanitize?"paragraph":"html",pre:"pre"===i[1]||"script"===i[1],text:i[0]});else if(t&&(i=this.rules.def.exec(e)))e=e.substring(i[0].length),this.tokens.links[i[1].toLowerCase()]={href:i[2],title:i[3]};else if(t&&(i=this.rules.table.exec(e))){for(e=e.substring(i[0].length),h={type:"table",header:i[1].replace(/^ *| *\| *$/g,"").split(/ *\| */),align:i[2].replace(/^ *|\| *$/g,"").split(/ *\| */),cells:i[3].replace(/(?: *\| *)?\n$/,"").split("\n")},u=0;u<h.align.length;u++)/^ *-+: *$/.test(h.align[u])?h.align[u]="right":/^ *:-+: *$/.test(h.align[u])?h.align[u]="center":/^ *:-+ *$/.test(h.align[u])?h.align[u]="left":h.align[u]=null;for(u=0;u<h.cells.length;u++)h.cells[u]=h.cells[u].replace(/^ *\| *| *\| *$/g,"").split(/ *\| */);this.tokens.push(h)}else if(t&&(i=this.rules.paragraph.exec(e)))e=e.substring(i[0].length),this.tokens.push({type:"paragraph",text:"\n"===i[1][i[1].length-1]?i[1].slice(0,-1):i[1]});else if(i=this.rules.text.exec(e))e=e.substring(i[0].length),this.tokens.push({type:"text",text:i[0]});else if(e)throw new Error("Infinite loop on byte: "+e.charCodeAt(0));return this.tokens};var a={escape:/^\\([\\`*{}\[\]()#+\-.!_>])/,autolink:/^<([^ >]+(@|:\/)[^ >]+)>/,url:r,tag:/^\x3c!--[\s\S]*?--\x3e|^<\/?\w+(?:"[^"]*"|'[^']*'|[^'">])*?>/,link:/^!?\[(inside)\]\(href\)/,reflink:/^!?\[(inside)\]\s*\[([^\]]*)\]/,nolink:/^!?\[((?:\[[^\]]*\]|[^\[\]])*)\]/,strong:/^__([\s\S]+?)__(?!_)|^\*\*([\s\S]+?)\*\*(?!\*)/,em:/^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,code:/^(`+)\s*([\s\S]*?[^`])\s*\1(?!`)/,br:/^ {2,}\n(?!\s*$)/,del:r,text:/^[\s\S]+?(?=[\\<!\[_*`]| {2,}\n|$)/};a._inside=/(?:\[[^\]]*\]|[^\]]|\](?=[^\[]*\]))*/,a._href=/\s*<?([^\s]*?)>?(?:\s+['"]([\s\S]*?)['"])?\s*/,a.link=i(a.link)("inside",a._inside)("href",a._href)(),a.reflink=i(a.reflink)("inside",a._inside)(),a.normal=l({},a),a.pedantic=l({},a.normal,{strong:/^__(?=\S)([\s\S]*?\S)__(?!_)|^\*\*(?=\S)([\s\S]*?\S)\*\*(?!\*)/,em:/^_(?=\S)([\s\S]*?\S)_(?!_)|^\*(?=\S)([\s\S]*?\S)\*(?!\*)/}),a.gfm=l({},a.normal,{escape:i(a.escape)("])","~|])")(),url:/^(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/,del:/^~~(?=\S)([\s\S]*?\S)~~/,text:i(a.text)("]|","~]|")("|","|https?://|")()}),a.breaks=l({},a.gfm,{br:i(a.br)("{2,}","*")(),text:i(a.gfm.text)("{2,}","*")()}),t.rules=a,t.output=function(e,s,n){var i=new t(s,n);return i.output(e)},t.prototype.output=function(e){for(var t,s,i,r,l="";e;)if(r=this.rules.escape.exec(e))e=e.substring(r[0].length),l+=r[1];else if(r=this.rules.autolink.exec(e))e=e.substring(r[0].length),"@"===r[2]?(s=":"===r[1][6]?this.mangle(r[1].substring(7)):this.mangle(r[1]),i=this.mangle("mailto:")+s):(s=n(r[1]),i=s),l+='<a href="'+i+'">'+s+"</a>";else if(r=this.rules.url.exec(e))e=e.substring(r[0].length),s=n(r[1]),i=s,l+='<a href="'+i+'">'+s+"</a>";else if(r=this.rules.tag.exec(e))e=e.substring(r[0].length),l+=this.options.sanitize?n(r[0]):r[0];else if(r=this.rules.link.exec(e))e=e.substring(r[0].length),l+=this.outputLink(r,{href:r[2],title:r[3]});else if((r=this.rules.reflink.exec(e))||(r=this.rules.nolink.exec(e))){if(e=e.substring(r[0].length),t=(r[2]||r[1]).replace(/\s+/g," "),t=this.links[t.toLowerCase()],!t||!t.href){l+=r[0][0],e=r[0].substring(1)+e;continue}l+=this.outputLink(r,t)}else if(r=this.rules.strong.exec(e))e=e.substring(r[0].length),l+="<strong>"+this.output(r[2]||r[1])+"</strong>";else if(r=this.rules.em.exec(e))e=e.substring(r[0].length),l+="<em>"+this.output(r[2]||r[1])+"</em>";else if(r=this.rules.code.exec(e))e=e.substring(r[0].length),l+="<code>"+n(r[2],!0)+"</code>";else if(r=this.rules.br.exec(e))e=e.substring(r[0].length),l+="<br>";else if(r=this.rules.del.exec(e))e=e.substring(r[0].length),l+="<del>"+this.output(r[1])+"</del>";else if(r=this.rules.text.exec(e))e=e.substring(r[0].length),l+=n(r[0]);else if(e)throw new Error("Infinite loop on byte: "+e.charCodeAt(0));return l},t.prototype.outputLink=function(e,t){return"!"!==e[0][0]?'<a href="'+n(t.href)+'"'+(t.title?' title="'+n(t.title)+'"':"")+">"+this.output(e[1])+"</a>":'<img src="'+n(t.href)+'" alt="'+n(e[1])+'"'+(t.title?' title="'+n(t.title)+'"':"")+">"},t.prototype.smartypants=function(e){return this.options.smartypants?e.replace(/--/g,"\u2014").replace(/'([^']*)'/g,"\u2018$1\u2019").replace(/"([^"]*)"/g,"\u201c$1\u201d").replace(/\.{3}/g,"\u2026"):e},t.prototype.mangle=function(e){for(var t,s="",n=e.length,i=0;i<n;i++)t=e.charCodeAt(i),Math.random()>.5&&(t="x"+t.toString(16)),s+="&#"+t+";";return s},s.parse=function(e,t){var n=new s(t);return n.parse(e)},s.prototype.parse=function(e){this.inline=new t(e.links,this.options),this.tokens=e.reverse();for(var s="";this.next();)s+=this.tok();return s},s.prototype.next=function(){return this.token=this.tokens.pop()},s.prototype.peek=function(){return this.tokens[this.tokens.length-1]||0},s.prototype.parseText=function(){for(var e=this.token.text;"text"===this.peek().type;)e+="\n"+this.next().text;return this.inline.output(e)},s.prototype.tok=function(){switch(this.token.type){case"space":return"";case"hr":return"<hr>\n";case"heading":return"<h"+this.token.depth+">"+this.inline.output(this.token.text)+"</h"+this.token.depth+">\n";case"code":if(this.options.highlight){var e=this.options.highlight(this.token.text,this.token.lang);null!=e&&e!==this.token.text&&(this.token.escaped=!0,this.token.text=e)}return this.token.escaped||(this.token.text=n(this.token.text,!0)),"<pre><code"+(this.token.lang?' class="'+this.options.langPrefix+this.token.lang+'"':"")+">"+this.token.text+"</code></pre>\n";case"table":var t,s,i,r,l,h="";for(h+="<thead>\n<tr>\n",s=0;s<this.token.header.length;s++)t=this.inline.output(this.token.header[s]),h+=this.token.align[s]?'<th align="'+this.token.align[s]+'">'+t+"</th>\n":"<th>"+t+"</th>\n";for(h+="</tr>\n</thead>\n",h+="<tbody>\n",s=0;s<this.token.cells.length;s++){for(i=this.token.cells[s],h+="<tr>\n",l=0;l<i.length;l++)r=this.inline.output(i[l]),h+=this.token.align[l]?'<td align="'+this.token.align[l]+'">'+r+"</td>\n":"<td>"+r+"</td>\n";h+="</tr>\n"}return h+="</tbody>\n","<table>\n"+h+"</table>\n";case"blockquote_start":for(var h="";"blockquote_end"!==this.next().type;)h+=this.tok();return"<blockquote>\n"+h+"</blockquote>\n";case"list_start":for(var o=this.token.ordered?"ol":"ul",h="";"list_end"!==this.next().type;)h+=this.tok();return"<"+o+">\n"+h+"</"+o+">\n";case"list_item_start":for(var h="";"list_item_end"!==this.next().type;)h+="text"===this.token.type?this.parseText():this.tok();return"<li>"+h+"</li>\n";case"loose_item_start":for(var h="";"list_item_end"!==this.next().type;)h+=this.tok();return"<li>"+h+"</li>\n";case"html":return this.token.pre||this.options.pedantic?this.token.text:this.inline.output(this.token.text);case"paragraph":return"<p>"+this.inline.output(this.token.text)+"</p>\n";case"text":return"<p>"+this.parseText()+"</p>\n"}},r.exec=r,h.options=h.setOptions=function(e){return l(h.defaults,e),h},h.defaults={gfm:!0,tables:!0,breaks:!1,pedantic:!1,sanitize:!1,smartLists:!1,silent:!1,highlight:null,langPrefix:""},h.Parser=s,h.parser=s.parse,h.Lexer=e,h.lexer=e.lex,h.InlineLexer=t,h.inlineLexer=t.output,h.parse=h,"object"==typeof exports?module.exports=h:"function"==typeof define&&define.amd?define(function(){return h}):this.marked=h}).call(function(){return this||("undefined"!=typeof window?window:global)}());