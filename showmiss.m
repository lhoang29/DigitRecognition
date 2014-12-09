function [] = showmiss(missim,misslab,testimages,testlabels,numshow,numpages)
    nummiss = nnz(missim);
    
    page = 1;
    showsize = floor(sqrt(numshow));
    for f=1:numshow:nummiss
        figure(floor(f/numshow) + 1);
        for m=f:min(nummiss,f+numshow-1)
            subplot(showsize,showsize,m-f+1);
            imshow(testimages(:,:,missim(m)));
            title(strcat(num2str(testlabels(missim(m))), ':', num2str(misslab(m))));
        end
        page = page + 1;
        if page > numpages
            break;
        end
    end

end